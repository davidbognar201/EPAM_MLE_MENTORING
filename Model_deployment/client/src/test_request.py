import requests
import pandas as pd
import datetime


dataframe = pd.read_csv("/client/src/data/dataset_wo_labels.csv")
payload = dataframe.to_dict(orient="records")


predictions = requests.post("http://batch-server:8000/predict-batch", json=payload).json()
predictions_df = pd.DataFrame.from_dict(predictions["result"])

dataframe = dataframe.join(predictions_df)
dataframe.to_csv("/client/src/data/dataset_w_labels.csv")

with open("/client/src/logs/request_log.txt", mode='a') as file:
    file.write('Predicted labels are sent by batch server at %s.\n' % 
               (datetime.datetime.now()))

