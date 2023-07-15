import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

bike_rental_df = pd.read_csv("Experiment_tracking/client/data/rentalData.csv")
bike_rental_df.rename(mapper={"cnt" : "target"}, inplace=True, axis=1)

target_feature = bike_rental_df["target"]
independent_features = bike_rental_df.drop(labels=["target", "casual", "registered", "instant", "dteday"], axis=1)

x_train, x_test, y_train, y_test = train_test_split(independent_features,
                                                    target_feature,
                                                    test_size=0.25,
                                                    random_state=42)

print("Shape of the train set: " + str(x_train.shape))
print("Shape of the test set: " + str(x_test.shape))

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

y_test = np.log(y_test)
y_train = np.log(y_train)


train_df = pd.DataFrame(data=x_train_scaled, columns=x_train.columns)
train_df["target"] = y_train

test_df = pd.DataFrame(data=x_test_scaled, columns=x_test.columns)
test_df["target"] = y_test


train_df.to_csv("Experiment_tracking/client/data/preprocessed/train_data.csv", index=False)
test_df.to_csv("Experiment_tracking/client/data/preprocessed/test_data.csv", index=False)

print(" ------------------ Preprocessing script finished ------------------ ")