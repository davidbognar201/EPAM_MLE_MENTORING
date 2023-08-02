import requests
import pandas as pd

dataframe = pd.read_csv("client/src/data/dataset_wo_labels.csv")
payload = dataframe.to_dict(orient="records")


predictions = requests.post("http://batch-server:8000/predict-batch", json=payload).json()
predictions_df = pd.DataFrame.from_dict(predictions["result"])
print(predictions_df)
dataframe = dataframe.join(predictions_df)
dataframe.to_csv("client/src/data/dataset_w_labels.csv")

