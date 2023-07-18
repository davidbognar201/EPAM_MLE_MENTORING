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
mlflow.create_experiment("KNN-Regressor")
mlflow.set_experiment("KNN-Regressor")

with mlflow.start_run(run_name="knnr-baseline"):
    baseline_model = KNeighborsRegressor()
    baseline_model.fit(train_x, train_y)
    predicted = baseline_model.predict(test_x)

    mlflow.log_metric("r2_score", round(r2_score(test_y, predicted), ndigits=2))
    mlflow.log_metric("mse", round(mean_squared_error(test_y, predicted), ndigits=2))
    mlflow.log_metric("mae", round(mean_absolute_error(test_y, predicted), ndigits=2))
 
    mlflow.log_param("n_neighbors", 5)
    mlflow.log_param("algorithm", "auto")
    mlflow.log_param("weights", "uniform")

    mlflow.sklearn.log_model(baseline_model, "KNN Regressor")
    print("Baseline finished")

############################### Hypertuned model ###############################

knn_regressor_param_grid = {
    "n_neighbors" : [3, 9, 11, 14],
    "algorithm" : ["ball_tree", "kd_tree", "auto"],
    "weights" : ["distance", "uniform"]
}

print("Experiment 2 - Hypeparameter search started")

knnr_tuned = GridSearchCV(estimator=KNeighborsRegressor(),
                                param_grid=knn_regressor_param_grid,
                                scoring="neg_mean_absolute_error",
                                cv=2,
                                verbose=3)
knnr_tuned.fit(train_x, train_y)

print("Experiment 2 - Hypeparameter search ended")

best_params = knnr_tuned.best_params_
print(best_params)

mlflow.set_experiment("KNN-Regressor")
with mlflow.start_run(run_name="knnr-hypertuned-1"):
    knnr_tuned = KNeighborsRegressor(**best_params)
    knnr_tuned.fit(train_x, train_y)
    predicted = knnr_tuned.predict(test_x)

    mlflow.log_metric("r2_score", round(r2_score(test_y, predicted), ndigits=2))
    mlflow.log_metric("mse", round(mean_squared_error(test_y, predicted), ndigits=2))
    mlflow.log_metric("mae", round(mean_absolute_error(test_y, predicted), ndigits=2))

    mlflow.log_param("n_neighbors", best_params["n_neighbors"])
    mlflow.log_param("algorithm", best_params["algorithm"])
    mlflow.log_param("weights", best_params["weights"])

    mlflow.sklearn.log_model(knnr_tuned, "KNN Regressor")
    print("Tuned finished")


print("Experiment 2 - Finished")
