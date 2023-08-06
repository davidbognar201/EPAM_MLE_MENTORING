import pandas as pd
import numpy as np
import sklearn
import pickle


def preprocessInput(inputs:dict):
    inputData = pd.DataFrame(data=inputs, index=[0])
    encoder, scaler = loadTransformers()
    numeric_features = inputData.select_dtypes("number").columns
    categorical_features = inputData.select_dtypes(object).columns

    numericProcessed = pd.DataFrame(data=scaler.transform(inputData[numeric_features]),
                                    columns=numeric_features)

    categoricalPreprocessed = pd.DataFrame(encoder.transform(inputData[categorical_features]).toarray(), 
                             columns = encoder.get_feature_names_out(categorical_features))
    
    final_data = numericProcessed.join(categoricalPreprocessed)
    return final_data

def loadTransformers():
    encoder = pickle.load(open('app/src/ml-assets/encoder.pkl', 'rb'))
    scaler = pickle.load(open('app/src/ml-assets/scaler.pkl', 'rb'))
    return encoder, scaler
    
def makePrediction(inputData:pd.DataFrame):
    processedInput = preprocessInput(inputData)
    model = pickle.load(open('app/src/ml-assets/model.pkl', 'rb'))
    
    probability = model.predict_proba(processedInput)
    result = model.predict(processedInput)
    prob_0 = round(probability[0][0]*100,2)
    prob_1 = round(probability[0][1]*100,2)
    if result == 0:
        return "Thyroid - Negative", prob_0
    else:
        return "Thyroid - Positive", prob_1

    