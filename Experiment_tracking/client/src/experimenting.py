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


test_data = pd.read_csv(r"Experiment_tracking\client\data\preprocessed\test_data.csv")
train_data = pd.read_csv(r"Experiment_tracking\client\data\preprocessed\train_data.csv")

print(train_data.columns)

train_x = train_data.drop(labels=['target'], axis=1)
test_x = test_data.drop(labels=['target'], axis=1)

train_y = train_data['target']
test_y = test_data['target']

mlflow.autolog()

random_forest_param_grid = {
    "criterion" : ["squared_error", "absolute_error", "friedman_mse", "poisson"],
    "max_features" : ["sqrt", "log2"],
    "n_estimators" : [100],
    
}

random_forest_model = cf.FitModelWithPredefinedParams(RandomForestRegressor(random_state=42), 
                                                    train_x = train_x,
                                                    train_y = train_y,
                                                    test_x = test_x,
                                                    param_grid = random_forest_param_grid,
                                                    scoring_metric = "neg_mean_absolute_error",
                                                    cv_value = 3,
                                                    verbose_value=3)

random_forest_predictions = random_forest_model.best_estimator_.predict(test_x)

cf.EvaluateModelPerformance(test_y, random_forest_predictions)