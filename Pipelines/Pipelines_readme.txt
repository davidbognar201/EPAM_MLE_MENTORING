Guide to reproduce HW:
  1, Copy AWS keys to the .env file
  2, Run docker-compose up and then the Airflow UI will be available at localhost:8080
  3, On the UI the ML-Pipeline DAG should be activated with the on/off switch next to the DAG name, after that the task will start running automatically 

HW description:

The main goal of the DAG is to fetch a database from an S3 storage, then it does the whole ML pipeline witch cleaning, preprocessing, train/test split and training. At the end it pushes the trained model to an S3 stroage with the metrics of the model.

The DAG consists of 14 PythonOperators and 1 BashOperator, so it runs a total of 15 tasks depending on the input.

Fetch tasks:
  1, fetch_data: Downloads data from a given S3 bucket
Data validation tasks:
  2, check_if_data_empty: Checks if the .csv file is empty, if yes, then raises an error
  3, check_if_data_is_valid: Checks if the fetched data has all the required columns
  4, check_if_data_contains_nan: Checks if it contains NaN values in any row, if yes then goes to task 5, if no, it continues with task 6
Data cleaning:
  5, drop_nan: If the task 04 founds any NaN values then this tasks drops every row with a NaN value, otherwise this task is skipped
Data Preprocessing:
  6, categorical_preprocess: One Hot Encoding of categorical variables
  7, train_test_split: Splits the preprocessed data to train and test set
  8, numerical_preprocess: Scales the continuous values separately in the train and test set 
Training the model:
  9, hyperparameter_tuning: Finds the best parameters using GridSearchCV and pushes them as XCOMs, so other tasks can pull these metrics
  10,train_model: Pulls the previously found parameters and trains a model on the train set, then pushes the model
  11, predict_on_test: Pulls the previously trained model and creates a prediction on the test set
Store artifacts:
  12, save_metrics: Saves the metrics on the test set as a .csv file
  13, serialize_model: Serializes the model with pickle and saves it as a .sav file
  14, push_to_remote: Pushes the metrics and the trained model to a remote S3 storage
Cleanup:
  15, clean_temp_files: Runs a shell script which clears all the temporary files which was created during the run


Airflow best practises applied:
  - I imported every package in the local scope of the task functions, so this way it gives better performance because airflow doesn't have to parse every import all the time, only when the function is called
  - Used XCOM to pass small amount of data between tasks, but the dataframes are not passed via XCOMs as they are not optimised to pass large data
  - Every task is as thin as possible, all of them mainly responsible for one task only
