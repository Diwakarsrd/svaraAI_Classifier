import requests
import json

def test_batch_prediction():
    """Test the batch prediction endpoint"""
    url = "http://127.0.0.1:8001/predict-batch"
    headers = {"Content-Type": "application/json"}
    
    # Test data
    data = [
        "Yes, I'd love to schedule a call tomorrow.",
        "I'm not interested in this product.",
        "Can you send me more information about the pricing?"
    ]
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_batch_prediction()