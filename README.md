# API Description

## API Usage

#### Endpoint
POST http://18.217.245.182/model/prediction

#### Request Body

- `data`: (array) a 28x28 grayscale image represented as a 2D array.

Example:

```python
{
    "data": [
        [10, 23, 59, 120, ... , 99, 55, 10],
        [34, 45, 68, 180, ... , 165, 70, 20],
        ...,
        [40, 65, 85, 140, ... , 175, 80, 30],
        [20, 35, 55, 120, ... , 110, 55, 10]
    ]
}
```

### Response

#### Response Body
`prediction`: (int) the predicted label for the input image.
```json
{
    "prediction": 7
}
```

####Status Codes
- `200 OK`: Successful prediction.
- `400 Bad Request`: Malformed request body.


## Example Usage

```python
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
```

## Dependencies
- `numpy`: : version 1.21.0 or later. Install via pip:
