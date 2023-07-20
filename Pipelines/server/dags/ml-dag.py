
from datetime import timedelta

import airflow
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator

FILENAME = 'rentalData.csv'
FILEPATH = 'dags/tmp_data/'
BUCKET_NAME = 'pipelines-module-data-storage'
VALID_COLUMNS = ["instant", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"]

CATEGORICAL_FEATURES = []
NUMERIC_FEATURES = []

def fetchBaseData():
    import boto3 

    resource = boto3.resource('s3')
    my_bucket = resource.Bucket(BUCKET_NAME)
    my_bucket.download_file(FILENAME, FILEPATH + FILENAME)

def checkEmptyData():
    import csv

    with open(FILEPATH + FILENAME) as csvfile:
            reader = csv.reader(csvfile)
            for i, _ in enumerate(reader):
                if i:  # found the second row
                    return True
    raise TypeError('Empty data file')

def checkDataValidity():
    import csv

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
    import pandas as pd

    data = pd.read_csv(FILEPATH + FILENAME)
    num_of_missing = data.isnull().any(axis=1).sum()
    if num_of_missing != 0:
        return "drop_nan"
    else:
        return "categorical_preprocess"
       

def dropRowsWithNullValues():
    import pandas as pd

    data = pd.read_csv(FILEPATH + FILENAME)
    num_of_missing = data.isnull().any(axis=1).sum()
    data.dropna(axis=0, inplace=True)
    print(str(num_of_missing) + " rows were dropped due to NaN values from " + FILENAME)
    data.to_csv(FILEPATH + FILENAME)
    print("Data file was replaced with cleaned data")


def categoricalPreprocessing():
    import pandas as pd
    return None

def traintestSplit():
    import pandas as pd
    from sklearn.model_selection import train_test_split

    return None

def numericPreprocessing():
    import pandas as pd
    from sklearn.model_selection import train_test_split

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
    task_id='drop_nan',
    python_callable=dropRowsWithNullValues,
    dag=dag)
drop_nan_task.set_upstream(data_nan_check)

categorical_preprocess = PythonOperator(
    task_id='categorical_preprocess',
    python_callable=categoricalPreprocessing,
    dag=dag)
categorical_preprocess.set_upstream(data_nan_check)
categorical_preprocess.set_upstream(drop_nan_task)

train_test_split = PythonOperator(
    task_id='train_test_split',
    python_callable=traintestSplit,
    dag=dag)
train_test_split.set_upstream(categorical_preprocess)

numerical_preprocess = PythonOperator(
    task_id='numerical_preprocess',
    python_callable=numericPreprocessing,
    dag=dag)
numerical_preprocess.set_upstream(train_test_split)

train_model = PythonOperator(
    task_id='train_model',
    python_callable=trainModel,
    dag=dag)
train_model.set_upstream(numerical_preprocess)


