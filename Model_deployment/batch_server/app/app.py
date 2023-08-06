from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
import uvicorn 
from typing import List
import pandas as pd
# own package
from deployment_utils.src.schemas import Input
from deployment_utils.src.preprocessing import DataPreprocess




app = FastAPI()

@app.get("/")
def root():
    return {"message":"Hello World"}

@app.post('/predict-batch')
async def predict_batch(inputs: List[Input]):
    input_df = pd.DataFrame([vars(s) for s in inputs])
    preprocess_ins = DataPreprocess(encoder_path="app/src/ml-assets/encoder.pkl",
                                    scaler_path="app/src/ml-assets/scaler.pkl",
                                    model_path="app/src/ml-assets/model.pkl")
    
    prediction = preprocess_ins.makePrediction(input_df)
    return {"result":prediction.to_dict(orient="records")}


