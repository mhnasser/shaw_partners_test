# API Description

## API Usage

#### Endpoint
POST http://18.217.245.182/model/prediction

#### Request Body

- `data`: (array) a 28x28 grayscale image represented as a 2D array.

Example:

```json
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
