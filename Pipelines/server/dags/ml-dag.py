
from datetime import timedelta

import airflow
from airflow import DAG
from airflow.models.xcom import XCom
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator

BASE_FILENAME = 'rentalData.csv'
ENCODED_FILENAME = 'encoded_data.csv'
TRAIN_FILENAME = "train_data.csv"
TEST_FILENAME = "test_data.csv"
TRAIN_PREPROCESSED_FILENAME = "train_data_preprocess.csv"
TEST_PREPROCESSED_FILENAME = "test_data_preprocess.csv"
FITTED_MODEL_FILENAME = "model.sav" 
METRICS_FILENAME = "metrics.csv"

TO_REMOTE_PATH = 'dags/to_remote/'
FILEPATH = 'dags/tmp_data/'
BUCKET_NAME = 'pipelines-module-data-storage'

VALID_COLUMNS = ["instant", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"]
CATEGORICAL_FEATURES = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
NUMERIC_FEATURES = ["temp", "atemp", "hum", "windspeed"]
TARGET = "cnt"
CV_PARAM_GRID = {
    "loss" : ['huber', 'quantile', 'absolute_error', 'squared_error'], 
    "criterion" : ["friedman_mse"],
    "n_estimators" : [500],
    "max_features" : [1, "log2", "sqrt"]
}
CROSS_VALIDATION_FOLDS = 2
SCORING_METRIC = "neg_mean_absolute_error" 
RANDOM_SEED = 42

def fetchBaseData():
    import boto3 

    resource = boto3.resource('s3')
    my_bucket = resource.Bucket(BUCKET_NAME)
    my_bucket.download_file(BASE_FILENAME, FILEPATH + BASE_FILENAME)


def checkEmptyData():
    import csv

    with open(FILEPATH + BASE_FILENAME) as csvfile:
            reader = csv.reader(csvfile)
            for i, _ in enumerate(reader):
                if i:  # found the second row
                    return True
    raise TypeError('Empty data file')

def checkDataValidity():
    import csv

    with open(FILEPATH + BASE_FILENAME, newline='') as f:
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

    data = pd.read_csv(FILEPATH + BASE_FILENAME)
    num_of_missing = data.isnull().any(axis=1).sum()
    if num_of_missing != 0:
        return "drop_nan"
    else:
        return "categorical_preprocess"
       

def dropRowsWithNullValues():
    import pandas as pd

    data = pd.read_csv(FILEPATH + BASE_FILENAME)
    num_of_missing = data.isnull().any(axis=1).sum()
    data.dropna(axis=0, inplace=True)
    print(str(num_of_missing) + " rows were dropped due to NaN values from " + BASE_FILENAME)
    data.to_csv(FILEPATH + BASE_FILENAME)
    print("Data file was replaced with cleaned data")


def categoricalPreprocessing():
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder

    data = pd.read_csv(FILEPATH + BASE_FILENAME)
    encoder = OneHotEncoder()
    encoded_data = pd.DataFrame(encoder.fit_transform(data[CATEGORICAL_FEATURES]).toarray(), columns=encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    final_df = data.drop(columns=CATEGORICAL_FEATURES, axis=1).join(encoded_data)
    final_df.drop(labels=final_df.columns[0:2], axis=1).to_csv(FILEPATH + ENCODED_FILENAME)


def traintestSplit():
    import pandas as pd
    from sklearn.model_selection import train_test_split

    data = pd.read_csv(FILEPATH + ENCODED_FILENAME)
    target_feature = data["cnt"]
    independent_features = data.drop(labels=["cnt", "casual", "registered", "dteday"], axis=1)

    try:
        x_train, x_test, y_train, y_test = train_test_split(independent_features,
                                                        target_feature,
                                                        test_size=0.2,
                                                        random_state=42)
    except:
        raise ValueError("Error has occured while splitting the data")
    
    print("Shape of the train set: " + str(x_train.shape))
    print("Shape of the test set: " + str(x_test.shape))

    train_data = pd.concat([x_train, y_train], axis=1)
    test_data = pd.concat([x_test, y_test], axis=1)

    try:
        train_data.to_csv(FILEPATH + TRAIN_FILENAME)
        test_data.to_csv(FILEPATH + TEST_FILENAME)
    except:
        raise IOError("Error has occured while exporting the data")
    
    print("Train test split succesfully saved")


def numericPreprocessing():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    train_data = pd.read_csv(FILEPATH + TRAIN_FILENAME)
    test_data = pd.read_csv(FILEPATH + TEST_FILENAME)

    scaler = StandardScaler()
    train_scaled = pd.DataFrame(data=scaler.fit_transform(train_data[NUMERIC_FEATURES]),
                                columns=NUMERIC_FEATURES)

    test_scaled = pd.DataFrame(data=scaler.transform(test_data[NUMERIC_FEATURES]),
                                columns=NUMERIC_FEATURES)

    train_target = np.log(train_data["cnt"])
    test_target = np.log(test_data["cnt"])

    train_preprocessed = pd.concat([train_data.drop(labels=NUMERIC_FEATURES, axis=1),
                                train_scaled,
                                train_target],
                                axis=1)
    test_preprocessed = pd.concat([test_data.drop(labels=NUMERIC_FEATURES, axis=1),
                                test_scaled,
                                test_target],
                                axis=1)
    
    train_preprocessed.to_csv(FILEPATH + TRAIN_PREPROCESSED_FILENAME)
    test_preprocessed.to_csv(FILEPATH + TEST_PREPROCESSED_FILENAME)
    

def hyperparameterTuning(ti,**kwargs):
    import pandas as pd
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import GradientBoostingRegressor
    train_data = pd.read_csv(FILEPATH + TRAIN_FILENAME)
    train_x = train_data.drop(labels=["cnt"], axis=1)
    train_y = train_data["cnt"]

    cv_ins = GridSearchCV(GradientBoostingRegressor(random_state=RANDOM_SEED),
                                 param_grid=CV_PARAM_GRID,
                                 cv=CROSS_VALIDATION_FOLDS,
                                 scoring=SCORING_METRIC,
                                 verbose=2)
    cv_ins.fit(train_x, train_y)
    print("Best parameters found")
    print(cv_ins.best_params_)
    ti.xcom_push(key='best_params', value=cv_ins.best_params_)

def trainModel(ti, **kwargs):
    import pandas as pd
    from sklearn.ensemble import GradientBoostingRegressor
    
    best_parameters = ti.xcom_pull(key='best_params', task_ids=["hyperparameter_tuning"])
    
    train_data = pd.read_csv(FILEPATH + TRAIN_FILENAME)
    train_x = train_data.drop(labels=["cnt"], axis=1)
    train_y = train_data["cnt"]

    gradBoostModel = GradientBoostingRegressor(**best_parameters[0])
    
    gradBoostModel.fit(train_x, train_y)
    ti.xcom_push(key='fitted_model', value=gradBoostModel)
    
def predictOnTestSet(ti, **kwargs):
    import pandas as pd
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

    test_data = pd.read_csv(FILEPATH + TEST_FILENAME)
    test_x = test_data.drop(labels=["cnt"], axis=1)
    test_y = test_data["cnt"]

    gradBoostModel =  ti.xcom_pull(key='fitted_model', task_ids=["train_model"])
    print(gradBoostModel)
    gradBoostModel = gradBoostModel[0]
    predictions = gradBoostModel.predict(test_x)

    test_metrics = {
        "mae": round(mean_absolute_error(test_y, predictions),ndigits=4),
        "mape": round(mean_absolute_percentage_error(test_y, predictions),ndigits=4),
        "mse": round(mean_squared_error(test_y, predictions),ndigits=4),
        "r2": round(r2_score(test_y, predictions),ndigits=4)
    }
    ti.xcom_push(key='metrics', value=test_metrics)

def serializeFittedModel(ti, **kwargs):
    import pickle
    fitted_model = ti.xcom_pull(key='fitted_model', task_ids=["train_model"])
    try:
        pickle.dump(fitted_model[0], open(TO_REMOTE_PATH + FITTED_MODEL_FILENAME, 'wb'))
    except:
        raise IOError("An error has occured while exporting the model file")
    
def saveMetrics(ti, **kwargs):
    import pandas as pd
    metrics = ti.xcom_pull(key='metrics', task_ids=["predict_on_test"])
    metrics_df = pd.DataFrame(data=metrics[0].items(), columns=["metric", "value"])
    metrics_df.to_csv(TO_REMOTE_PATH + METRICS_FILENAME)

def pushArtifactsToRemote(ti, **kwargs):
    import boto3

    resource = boto3.resource('s3')
    resource.meta.client.upload_file(Filename=TO_REMOTE_PATH + FITTED_MODEL_FILENAME,
                                 Bucket=BUCKET_NAME,
                                 Key=FITTED_MODEL_FILENAME)
    resource.meta.client.upload_file(Filename=TO_REMOTE_PATH + METRICS_FILENAME,
                                 Bucket=BUCKET_NAME,
                                 Key=METRICS_FILENAME)



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
categorical_preprocess.set_upstream([data_nan_check,drop_nan_task])

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

hyperparam_tune = PythonOperator(
    task_id='hyperparameter_tuning',
    python_callable=hyperparameterTuning,
    dag=dag,
    provide_context=True)
hyperparam_tune.set_upstream(numerical_preprocess)

train_model = PythonOperator(
    task_id='train_model',
    python_callable=trainModel,
    dag=dag,
    provide_context=True)
train_model.set_upstream(hyperparam_tune)

predict_on_test = PythonOperator(
    task_id='predict_on_test',
    python_callable=predictOnTestSet,
    dag=dag,
    provide_context=True)
predict_on_test.set_upstream(train_model)

save_metrics = PythonOperator(
    task_id='save_metrics',
    python_callable=saveMetrics,
    dag=dag,
    provide_context=True)
save_metrics.set_upstream(predict_on_test)

serializeModel = PythonOperator(
    task_id='serialize_model',
    python_callable=serializeFittedModel,
    dag=dag,
    provide_context=True)
serializeModel.set_upstream(predict_on_test)

push_to_remote = PythonOperator(
    task_id='push_to_remote',
    python_callable=pushArtifactsToRemote,
    dag=dag,
    provide_context=True)
push_to_remote.set_upstream(save_metrics)
push_to_remote.set_upstream(serializeModel)

clean_temp_files = BashOperator(
    task_id='clean_temp_files',
    bash_command='scripts/cleanTempFolder.sh',
    dag=dag)
clean_temp_files.set_upstream(push_to_remote)

