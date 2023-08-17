import csv
import io
import pandas as pd
import requests

files = {'file': ("payload.csv", open("payload.csv", 'rb'), 'text/csv')}
url = "http://localhost:8080/invocations"
header = {'Content-type': 'text/csv'}
files = {'upload_file': open("payload.csv",'rb')}
payload = pd.read_csv("payload.csv")
payload = payload.to_json(orient="split")

r = requests.post(url, data=payload)

print(r.text)