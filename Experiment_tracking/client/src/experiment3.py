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

train_x = train_data.drop(labels=['target'], axis=1)
test_x = test_data.drop(labels=['target'], axis=1)

train_y = train_data['target']
test_y = test_data['target']

RANDOM_SEED = 42
############################### Baseline model ###############################
mlflow.create_experiment("Gradient Boosting Regressor")
mlflow.set_experiment("Gradient Boosting Regressor")

with mlflow.start_run(run_name="grad-boost-baseline"):
    baseline_model = GradientBoostingRegressor(random_state=RANDOM_SEED)
    baseline_model.fit(train_x, train_y)
    predicted = baseline_model.predict(test_x)

    mlflow.log_metric("r2_score", round(r2_score(test_y, predicted), ndigits=2))
    mlflow.log_metric("mse", round(mean_squared_error(test_y, predicted), ndigits=2))
    mlflow.log_metric("mae", round(mean_absolute_error(test_y, predicted), ndigits=2))
 
    mlflow.log_param("loss", "squared_error")
    mlflow.log_param("criterion", "friedman_mse")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_features", None)

    mlflow.sklearn.log_model(baseline_model, "Gradient_Boosting_Regressor")
    print("Baseline finished")

############################### Hypertuned model ###############################

boosting_regression_param_grid = {
    "loss" : ['huber', 'quantile', 'squared_error'], 
    "criterion" : ["friedman_mse"],
    "n_estimators" : [200, 300],
    "max_features" : [1, "log2", "sqrt"]
}

print("Experiment 3 - Hypeparameter search started")

grad_boost_tuned = GridSearchCV(estimator=GradientBoostingRegressor(random_state=RANDOM_SEED),
                                param_grid=boosting_regression_param_grid,
                                scoring="neg_mean_absolute_error",
                                cv=2,
                                verbose=3)
grad_boost_tuned.fit(train_x, train_y)

print("Experiment 3 - Hypeparameter search ended")

best_params = grad_boost_tuned.best_params_
print(best_params)

mlflow.set_experiment("Gradient Boosting Regressor")
with mlflow.start_run(run_name="grad-boost-hypertuned-1"):
    grad_boost_tuned = GradientBoostingRegressor(**best_params, random_state=RANDOM_SEED)
    grad_boost_tuned.fit(train_x, train_y)
    predicted = grad_boost_tuned.predict(test_x)

    mlflow.log_metric("r2_score", round(r2_score(test_y, predicted), ndigits=2))
    mlflow.log_metric("mse", round(mean_squared_error(test_y, predicted), ndigits=2))
    mlflow.log_metric("mae", round(mean_absolute_error(test_y, predicted), ndigits=2))

    mlflow.log_param("loss", best_params["loss"])
    mlflow.log_param("criterion", best_params["criterion"])
    mlflow.log_param("n_estimators", best_params["n_estimators"])
    mlflow.log_param("max_features", best_params["max_features"])

    mlflow.sklearn.log_model(grad_boost_tuned, "Gradient_Boosting_Regressor")
    print("Tuned finished")


print("Experiment 3 - Finished")
