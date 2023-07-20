import boto3
from datetime import timedelta
import pandas as pd
import csv

import airflow
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator

FILENAME = 'rentalData.csv'
FILEPATH = '/data/'
BUCKET_NAME = 'pipelines-module-data-storage'
VALID_COLUMNS = ["instant", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"]

CATEGORICAL_FEATURES = []
NUMERIC_FEATURES = []

def fetchBaseData():
    try:
        s3 = boto3.client('s3')
        bucket=BUCKET_NAME
        target_file=FILENAME
        s3.download_file(bucket,target_file, FILEPATH + FILENAME)
    except:
        raise IOError("Failed to fetch data from S3 bucket - " + BUCKET_NAME)

def checkEmptyData():
    with open(FILEPATH + FILENAME) as csvfile:
            reader = csv.reader(csvfile)
            for i, _ in enumerate(reader):
                if i:  # found the second row
                    return True
    raise TypeError('Empty data file')

def checkDataValidity():
    with open(FILEPATH + FILENAME, newline='') as f:
        reader = csv.reader(f)
        input_header = next(reader)  
        if len(input_header) == len(VALID_COLUMNS):
            contained = [x in input_header for x in VALID_COLUMNS]
            if False not in contained:
                return True
            else:
                raise TypeError("Undefined column(s) in input")
        else:
            raise TypeError("Amount of columns is not identical to template (expected " + len(VALID_COLUMNS) + ", got " + len(input_header) + " instead)")
        

def checkNullValues():
    data = pd.read_csv(FILEPATH + FILENAME)
    num_of_missing = data.isnull().any(axis=1).sum()
    if num_of_missing != 0:
        return "drop_nan_task"
    else:
        return None
       

def dropRowsWithNullValues():
    data = pd.read_csv(FILEPATH + FILENAME)
    num_of_missing = data.isnull().any(axis=1).sum()
    data.dropna(axis=0, inplace=True)
    print(num_of_missing + " rows were dropped due to NaN values from " + FILENAME)
    data.to_csv(FILEPATH + FILENAME)
    print("Data file was replaced with cleaned data")
    return None

def categoricalPreprocessing():
    return None

def traintestSplit():
    return None

def numericPreprocessing():
    return None

def trainModel():
    return None

def createArtifacts():
    return None

def saveArtifactsToRemote():
    return None

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': airflow.utils.dates.days_ago(2),
    'email': ['david.bognar@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'adhoc':False,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'trigger_rule': u'all_success'
}

dag = DAG(
    'ML-pipeline',
    default_args=default_args,
    description='Fetch data from AWS then train model and upload model to storage',
    schedule_interval=timedelta(days=1),
)

fetch_tweets = PythonOperator(
    task_id='fetch_data',
    python_callable=fetchBaseData,
    dag=dag)

empty_data_check = PythonOperator(
    task_id='check_if_data_empty',
    python_callable=checkEmptyData,
    dag=dag)

empty_data_check.set_upstream(fetch_tweets)

data_validity_check = PythonOperator(
    task_id='check_if_data_is_valid',
    python_callable=checkDataValidity,
    dag=dag)

data_validity_check.set_upstream(empty_data_check)

data_nan_check = PythonOperator(
    task_id='check_if_data_contains_nan',
    python_callable=checkNullValues,
    dag=dag)

data_nan_check.set_upstream(data_validity_check)

drop_nan_task = PythonOperator(
    task_id='drop_nan_task',
    python_callable=dropRowsWithNullValues,
    dag=dag)

drop_nan_task.set_upstream(data_nan_check)


