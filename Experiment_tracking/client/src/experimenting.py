import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import customFunctions as cf
import mlflow

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split,  cross_val_score, GridSearchCV, RepeatedKFold
from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

import pylab 
import scipy.stats as stats


test_data = pd.read_csv("/client/data/preprocessed/test_data.csv")
train_data = pd.read_csv("/client/data/preprocessed/train_data.csv")

print(train_data.columns)

train_x = train_data.drop(labels=['target'], axis=1)
test_x = test_data.drop(labels=['target'], axis=1)

train_y = train_data['target']
test_y = test_data['target']

criterion = "friedman_mse"
max_features = "sqrt"
n_est = 100
RANDOM_SEED = 42
with mlflow.start_run():

    mlflow.log_param("criterion", criterion)
    mlflow.log_param("max_features", max_features)
    mlflow.log_param("n_estimators", n_est)



    rnd_forest_model = RandomForestClassifier(random_state=42,
                                             criterion=criterion,
                                             max_features=max_features,
                                             n_estimators=n_est)
    
    rnd_forest_model.fit(train_x, train_y)

    prediction = rnd_forest_model.predict(test_x)

    mlflow.log_metric("R2", r2_score(test_y, prediction))
    mlflow.log_metric("MAE", mean_absolute_error(test_y, prediction))
    mlflow.log_metric("MAPE", mean_absolute_percentage_error(test_y, prediction))