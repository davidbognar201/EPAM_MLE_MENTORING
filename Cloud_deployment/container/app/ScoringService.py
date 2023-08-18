import os
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

prefix = "/opt/ml/"
model_path = os.path.join(prefix)

CATEGORICAL_FEATURES = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
NUMERIC_FEATURES = ["temp", "atemp", "hum", "windspeed"]



class ScoringService(object):
    model = None  # Where we keep the model when it's loaded
    scaler = None
    encoder = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            with open(os.path.join(model_path, "model.pkl"), "rb") as inp:
                cls.model = pickle.load(inp)
        return cls.model
    
    @classmethod
    def get_preprocessing_utils(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.scaler == None:
            with open(os.path.join(model_path, "scaler.pkl"), "rb") as inp:
                cls.scaler = pickle.load(inp)

        if cls.encoder == None:
            with open(os.path.join(model_path, "encoder.pkl"), "rb") as inp:
                cls.encoder = pickle.load(inp)

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        cls.get_preprocessing_utils()
        input_processed = cls.preprocess_input(input)
        return clf.predict(input_processed)
    @classmethod
    def preprocess_input(cls, input:pd.DataFrame):
        input = input[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        input[CATEGORICAL_FEATURES] = input[CATEGORICAL_FEATURES].astype(int)
        input_scaled = pd.DataFrame(data=cls.scaler.transform(input[NUMERIC_FEATURES]),
                                columns=NUMERIC_FEATURES)
        input_encoded = pd.DataFrame(cls.encoder.transform(input[CATEGORICAL_FEATURES]).toarray(), 
                                     columns=cls.encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        input_preprocessed = pd.concat([input.drop(labels=NUMERIC_FEATURES + CATEGORICAL_FEATURES, axis=1),
                                input_scaled,
                                input_encoded],
                                axis=1)
        return input_preprocessed