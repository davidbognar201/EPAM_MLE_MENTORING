import pandas as pd
import preprocessFunctions as pf
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, roc_auc_score, confusion_matrix
import pickle

test_data = pd.read_csv(filepath_or_buffer = r"data/bike_rental_test_data.csv")
test_data.rename(mapper={"cnt" : "target"}, inplace=True, axis=1)

train_data = pd.read_csv(filepath_or_buffer = r"data/bike_rental_train_data.csv")
train_data.rename(mapper={"cnt" : "target"}, inplace=True, axis=1)

test_x, test_y = pf.PreprocessTestData(train_data, test_data)

model = pickle.load(open(r"trained_model/GradBoosting_Regressor_trained.sav", 'rb'))
predictions = model.predict(test_x)

print('----------------- Results on the test set -----------------')
print(f'R2 score: {round(r2_score(predictions, test_y),3)}')
print(f'MSE score: {round(mean_squared_error(predictions, test_y),3)}')
print(f'MAE score: {round(mean_absolute_error(predictions, test_y),3)}')
print(f'MAPE score: {round(mean_absolute_percentage_error(predictions, test_y),3)}')