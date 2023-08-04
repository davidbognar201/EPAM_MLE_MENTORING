import requests
import datetime
#print(requests.get("http://batch-server:8000/").json())

cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('JOB RUN AT : ',cur_time)