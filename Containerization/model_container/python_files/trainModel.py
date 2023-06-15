import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import preprocessFunctions as pf
import pickle



train_data = pd.read_csv(filepath_or_buffer = r"/model/data/bike_rental_train_data.csv")
train_data.rename(mapper={"cnt" : "target"}, inplace=True, axis=1)

train_x, train_y = pf.PreprocessTrainData(train_data)

rand_seed = 46
model_params = {
       "loss" : ['huber', 'quantile', 'absolute_error', 'squared_error'], 
        "criterion" : ["friedman_mse"],
        "n_estimators" : [1000],
        "max_features" : [1, "log2", "sqrt"]
}

model_gridsearch = GridSearchCV(GradientBoostingRegressor(random_state = rand_seed),
                                 param_grid = model_params,
                                 cv = 5,
                                 scoring = "neg_mean_absolute_error",
                                 verbose = 3,
                                 refit = True)

model_gridsearch.fit(train_x, train_y)
print("Training done")

# Saving the trained model
pickle.dump(model_gridsearch.best_estimator_, open("/model/model_iterations/GradBoosting_Regressor_trained.sav", 'wb'))


    
