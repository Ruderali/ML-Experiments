# app.py
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize Flask app
app = Flask(__name__)

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
model_path = 'results/checkpoint-2184'
model = model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#The dataset the model was trained was sigmoid normalized so we need to perform logistic sigmoid transformation to properly interpret the outputs
def isn(normalized_value):
    mu = 17.009805789666544
    alpha = 107.46853536029518
    return mu - alpha * np.log(1 / normalized_value - 1)

#This needs to be further dialed in as virtually all predictions are considered 'Medium'
#However in practice, the intended application would combine the confidence (i.e std) with the Δ between actual and predicted to inform QA processes
#This is mainly just to make the numbers more friendly
def intConfidence(std_dev):
    if std_dev > 0.04:
        return "Low" 
    elif 0.02 < std_dev <= 0.04:
        return "Medium" 
    else:
        return "High"

# As the base model provides fairly varied predictions so we perform a Monte Carlo Dropout to generate both an uncertainty value and a more accurate prediction 
# The cost for this is 200x the processing time (num_samples) however given the intended use has text between 5-10 words this is not an issue
def predict_with_uncertainty(model, tokenizer, text, num_samples=200):
    model.train()  # Enable dropout during inference
    predictions = []

    for _ in range(num_samples):
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            output = model(**inputs)
        
        predictions.append(output.logits.squeeze().item())

    mean_prediction = torch.mean(torch.tensor(predictions))
    std_prediction = torch.std(torch.tensor(predictions))

    confidence = intConfidence(std_prediction.item())
    norm = isn(mean_prediction.item())
    return norm, confidence

# Routes
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if text:
        norm, confidence = predict_with_uncertainty(model, tokenizer, text)
        return jsonify({'norm': norm, 'confidence': confidence})
    else:
        return jsonify({'error': 'No text provided'}), 400

@app.route('/')
def index():
    return send_from_directory(os.getcwd(), 'predict.html')

if __name__ == '__main__':
    app.run(debug=True)
