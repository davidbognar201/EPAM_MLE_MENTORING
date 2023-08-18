import csv
import io
import pandas as pd
import requests


url = "http://localhost:8080/invocations"

payload = pd.read_csv("payload.csv")
payload = payload.to_json(orient="split")

r = requests.post(url, data=payload)

print(r.text)

r2 = requests.get("http://localhost:8080/ping")
print(r2)