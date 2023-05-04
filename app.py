import sys
sys.path.append('001_modules')
from trainning_tools import *
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from torchvision.transforms import ToTensor

app = FastAPI()

class ImageInput(BaseModel):
    data: List[List[int]]

def load_model(model_path="02_models/mohamad_model.pth"):
    model, _, _ = get_model()
    model.load_state_dict(torch.load(model_path), 
                          map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()

@app.post("/model/prediction")
async def predict(input: ImageInput):
    input_image = torch.tensor(input.data, dtype=torch.float32)
    if input_image.shape != (28, 28):
        raise HTTPException(status_code=400, detail="Input image must be 28x28.")
    
    input_image = input_image.unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(input_image)
        max_values, argmaxes = prediction.max(-1)
        return {"class": argmaxes.item()}