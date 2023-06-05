from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def PreprocessTrainData(dataframe:pd.DataFrame):
    train_y = dataframe["target"]
    train_x = dataframe.drop(labels=["target", "casual", "registered", "instant", "dteday"], axis=1)

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    train_y = np.log(train_y)
    return train_x, train_y

def PreprocessTestData(train_df:pd.DataFrame, test_df:pd.DataFrame):
    train_x = train_df.drop(labels=["target", "casual", "registered", "instant", "dteday"], axis=1)
    test_y = test_df["target"]
    test_x = test_df.drop(labels=["target", "casual", "registered", "instant", "dteday"], axis=1)

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    test_y = np.log(test_y)
    return test_x, test_y