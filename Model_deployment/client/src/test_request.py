import requests
import pandas as pd

dataframe = pd.read_csv("src/data/dataset_wo_labels.csv")
payload = dataframe.head(2).to_dict(orient="records")


predictions = requests.post("http://server:8000/predict-batch", json=payload)

dataframe["prediction"] = predictions
dataframe.to_csv("src/data/dataset_w_labels.csv")
