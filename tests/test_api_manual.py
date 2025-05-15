import requests
import json
import os

def test_health():
    """Test the health endpoint"""
    response = requests.get("http://localhost:5000/health")
    print(f"Health check response: {response.status_code}")
    print(response.json())

def test_predict():
    """Test the prediction endpoint"""
    # Load test data from dados.json
    with open("../src/dados.json", "r") as f:
        dados = json.load(f)

    # Send POST request
    response = requests.post("http://localhost:5000/prever", json=dados)
    print(f"Prediction response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    # Run the API in a separate terminal first:
    # python src/app.py
    test_health()
    test_predict()