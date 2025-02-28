from flask import Flask, request, jsonify
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from flask_cors import CORS  # Allow frontend to communicate with backend
import praw
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
from dotenv import load_dotenv

# Load environment variables (for Reddit API credentials)
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})



# Model paths
FACT_CHECK_MODEL_PATH = "fact_checking_model.pkl"
SENTIMENT_MODEL_PATH = "sentiment_model.pkl"

# Function to load the fact-checking model
def load_fact_checking_model():
    try:
        model_data = joblib.load(FACT_CHECK_MODEL_PATH)

        tokenizer_name = model_data["tokenizer_name"]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        model = AutoModelForSequenceClassification.from_pretrained(tokenizer_name)
        model.load_state_dict(model_data["model_state_dict"])
        model.eval()

        print("✅ Fact-checking model loaded successfully!")
        return tokenizer, model
    except Exception as e:
        print(f"❌ Error loading fact-checking model: {str(e)}")
        return None, None

# Function to load sentiment analysis model
def load_sentiment_model():
    try:
        sentiment_model = joblib.load(SENTIMENT_MODEL_PATH)
        print("✅ Custom sentiment model loaded successfully!")
        return sentiment_model
    except:
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')

        sentiment_analyzer = SentimentIntensityAnalyzer()
        print("✅ NLTK VADER sentiment analyzer loaded as fallback!")
        return sentiment_analyzer

# Function to initialize Reddit API client
def init_reddit_client():
    try:
        reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent="sentiment-analyzer-app/1.0"
        )
        print("✅ Reddit API client initialized successfully!")
        return reddit
    except Exception as e:
        print(f"❌ Error initializing Reddit client: {str(e)}")
        return None

# Load both models when the server starts
fact_tokenizer, fact_model = load_fact_checking_model()
sentiment_model = load_sentiment_model()
reddit_client = init_reddit_client()

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for fake news detection.
    """
    if not fact_model or not fact_tokenizer:
        return jsonify({"error": "Fact-checking model not available"}), 500

    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Process text with the model
    inputs = fact_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = fact_model(**inputs)

    probabilities = softmax(outputs.logits, dim=-1)
    labels = ["Contradiction", "Neutral", "Entailment"]
    prediction = labels[torch.argmax(probabilities).item()]
    confidence = torch.max(probabilities).item()

    return jsonify({"prediction": prediction, "confidence": round(confidence, 4)})

@app.route('/sentiment', methods=['GET'])
def analyze_reddit_sentiment():
    """
    API endpoint to analyze sentiment of posts from a subreddit.
    """
    if not reddit_client:
        return jsonify({"error": "Reddit API client not initialized"}), 500

    subreddit_name = request.args.get('subreddit')
    if not subreddit_name:
        return jsonify({"error": "No subreddit provided"}), 400

    try:
        subreddit = reddit_client.subreddit(subreddit_name)
        posts = []

        for post in subreddit.hot(limit=25):
            # Analyze sentiment of post title
            if isinstance(sentiment_model, SentimentIntensityAnalyzer):
                sentiment_scores = sentiment_model.polarity_scores(post.title)
                compound_score = sentiment_scores['compound']
                
                if compound_score >= 0.05:
                    sentiment = "Positive"
                elif compound_score <= -0.05:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"
            else:
                # Using custom model
                try:
                    prediction = sentiment_model.predict([post.title])[0]
                    sentiment = prediction
                except Exception as e:
                    sentiment = "Error"
                    print(f"❌ Sentiment model error: {str(e)}")
                compound_score = 0  # Replace with actual score if available
            
            posts.append({
                "Title": post.title,
                "Score": post.score,
                "Sentiment": sentiment,
                "SentimentScore": compound_score,
                "URL": post.url,
                "NumComments": post.num_comments
            })
        
        return jsonify(posts)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
