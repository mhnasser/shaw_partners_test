import requests
import numpy as np

# Replace with the URL where your API is hosted (for local testing, use "http://localhost:8000")
api_url = "http://18.217.245.182/model/prediction"

# Create a random 28x28 image (replace with your actual input)
image_data = np.random.randint(0, 256, size=(28, 28)).tolist()

response = requests.post(api_url, json={"data": image_data})

if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Error:", response.text)