from flask import Flask, request, jsonify
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from flask_cors import CORS  # Allow frontend to communicate with backend

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

MODEL_PATH = "fact_checking_model.pkl"

def load_saved_model(model_path=MODEL_PATH):
    """
    Load the saved model and tokenizer.
    """
    model_data = joblib.load(model_path)

    # Load tokenizer using model name
    tokenizer_name = model_data["tokenizer_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load model and state dict
    model = AutoModelForSequenceClassification.from_pretrained(tokenizer_name)
    model.load_state_dict(model_data["model_state_dict"])
    model.eval()  # Set model to evaluation mode
    
    print("✅ Model and tokenizer loaded successfully!")
    return tokenizer, model

tokenizer, model = load_saved_model()  # Load model when the server starts

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to receive text, process it with the model, and return a prediction.
    """
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Process text with the model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = softmax(outputs.logits, dim=-1)
    labels = ["Contradiction", "Neutral", "Entailment"]
    prediction = labels[torch.argmax(probabilities).item()]
    confidence = torch.max(probabilities).item()

    return jsonify({"prediction": prediction, "confidence": round(confidence, 4)})

if __name__ == "__main__":
    app.run(debug=True)
