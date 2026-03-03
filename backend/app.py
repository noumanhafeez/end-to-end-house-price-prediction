# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)
model = joblib.load('./artifacts/house_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expect a JSON input
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return jsonify({'predicted_price': round(prediction, 2)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050)