
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn 
from typing import Optional, List
from src.schema import Input
import pandas as pd

app = FastAPI()

@app.get("/")
def root():
    return {"message":"Hello World"}

@app.post('/predict-batch')
async def predict_batch(inputs: List[Input]):
    input_df = pd.DataFrame([vars(s) for s in inputs])
    return {"OK":"!!!"}


