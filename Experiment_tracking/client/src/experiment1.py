import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import customFunctions as cf
import mlflow
import pickle

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split,  cross_val_score, GridSearchCV, RepeatedKFold
from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

import pylab 
import scipy.stats as stats

test_data = pd.read_csv("/client/data/preprocessed/test_data.csv")
train_data = pd.read_csv("/client/data/preprocessed/train_data.csv")

#test_data = pd.read_csv("client/data/preprocessed/test_data.csv")
#rain_data = pd.read_csv("client/data/preprocessed/train_data.csv")

train_x = train_data.drop(labels=['target'], axis=1)
test_x = test_data.drop(labels=['target'], axis=1)

train_y = train_data['target']
test_y = test_data['target']

RANDOM_SEED = 42
############################### Baseline model ###############################
mlflow.create_experiment("Random Forest Regressor")
mlflow.set_experiment("Random Forest Regressor")

with mlflow.start_run(run_name="baseline"):
    baseline_model = RandomForestRegressor(random_state=RANDOM_SEED)
    baseline_model.fit(train_x, train_y)
    predicted = baseline_model.predict(test_x)

    mlflow.log_metric("r2_score", round(r2_score(test_y, predicted), ndigits=2))
    mlflow.log_metric("mse", round(mean_squared_error(test_y, predicted), ndigits=2))
    mlflow.log_metric("mae", round(mean_absolute_error(test_y, predicted), ndigits=2))
 
    mlflow.log_param("max_features", 1.0)
    mlflow.log_param("criterion", "squared_error")
    mlflow.log_param("n_estimators", 100)

    mlflow.sklearn.log_model(baseline_model, "Random Forest Regressor")
    print("Baseline finished")

############################### Hypertuned model ###############################

param_space = {
    "max_features" : ["sqrt", "log2"],
    "criterion" : ["squared_error", "absolute_error", "friedman_mse", "poisson"],
    "n_estimators" : [100, 150]
}

print("Experiment1 - Hypeparameter search started")

rnd_forest_cv = GridSearchCV(estimator=RandomForestRegressor(random_state=RANDOM_SEED),
                                param_grid=param_space,
                                scoring="neg_mean_absolute_error",
                                cv=2,
                                verbose=3)
rnd_forest_cv.fit(train_x, train_y)

print("Experiment1 - Hypeparameter search ended")

best_params = rnd_forest_cv.best_params_
print(best_params)

mlflow.set_experiment("Random Forest Regressor")
with mlflow.start_run(run_name="rfr-hypertuned-1"):
    tuned_model = RandomForestRegressor(**best_params, random_state=RANDOM_SEED)
    tuned_model.fit(train_x, train_y)
    predicted = tuned_model.predict(test_x)

    mlflow.log_metric("r2_score", round(r2_score(test_y, predicted), ndigits=2))
    mlflow.log_metric("mse", round(mean_squared_error(test_y, predicted), ndigits=2))
    mlflow.log_metric("mae", round(mean_absolute_error(test_y, predicted), ndigits=2))
    mlflow.log_param("max_features", best_params["max_features"])
    mlflow.log_param("criterion", best_params["criterion"])
    mlflow.log_param("n_estimators", best_params["n_estimators"])
    mlflow.sklearn.log_model(tuned_model, "Random Forest Regressor")
    print("Tuned finished")


print("Experiment 1 - Finished")
