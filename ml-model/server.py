from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model_path = 'models/logistic_regression_model.pkl'
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the features from the request
    data = request.json
    features = [data['feature1'], data['feature2']]  # Adjust according to your features

    # Convert to DataFrame
    features_df = pd.DataFrame([features], columns=['feature1', 'feature2'])  # Adjust columns as needed

    # Make prediction
    prediction = model.predict(features_df)
    
    # Return the prediction result
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(port=5001)  # Run Flask on port 5001
