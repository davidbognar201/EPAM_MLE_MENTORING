from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn 
from typing import Optional, List
from app.src.schema import Input
from app.src.modules import makePrediction
import pandas as pd

app = FastAPI()

@app.get("/")
def root():
    return {"message":"Hello World"}

@app.post('/predict-batch')
async def predict_batch(inputs: List[Input]):
    input_df = pd.DataFrame([vars(s) for s in inputs])
    prediction = makePrediction(inputData=input_df,
                                encoder_path="app/src/ml-assets/encoder.pkl",
                                model_path="app/src/ml-assets/model.pkl",
                                scaler_path="app/src/ml-assets/scaler.pkl")
    return {"result":prediction.to_dict(orient="records")}


