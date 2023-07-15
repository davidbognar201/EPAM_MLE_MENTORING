import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV



from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

import pylab 
import scipy.stats as stats


def FitModelWithPredefinedParams(modelInstance, train_x, train_y:np.array, test_x, param_grid:dict, scoring_metric:str, cv_value:int, verbose_value=1):
    modelInstance = GridSearchCV(modelInstance,
                                 param_grid=param_grid,
                                 cv=cv_value,
                                 scoring=scoring_metric,
                                 verbose=verbose_value,
                                 refit=True)
    modelInstance.fit(train_x, train_y)
    return modelInstance

def EvaluateModelPerformance(predicted_y, true_y):
    print(f'R2 score: {r2_score(true_y, predicted_y)}')
    print(f'MSE score: {mean_squared_error(true_y, predicted_y)}')
    print(f'MAE score: {mean_absolute_error(true_y, predicted_y)}')
    print(f'MAPE score: {mean_absolute_percentage_error(true_y, predicted_y)}')

def GenerateDataFrameWithTrueAndPredictedValues():
    result_df = pd.DataFrame(columns=["true_value", "predicted_value", "error"])
    return result_df

def CheckNormalityOfResiduals(true_y_values, predicted_y_values, sig=0.05, qq_plot=True):
    residuals = true_y_values - predicted_y_values
    kolmogorov_test_result = stats.normaltest(residuals)
    if qq_plot:
        stats.probplot(residuals, dist="norm", plot=pylab)
    if kolmogorov_test_result.pvalue >= sig:
        return str(kolmogorov_test_result.pvalue) + " >= " + str(sig) + " -> Null-hypothesis is accepted, the residuals are normally distributed"
    else: return str(kolmogorov_test_result.pvalue) + " < " + str(sig) + " -> Null-hypothesis is rejected, the residuals are not normally distributed"
    