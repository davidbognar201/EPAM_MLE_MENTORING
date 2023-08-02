import pandas as pd
import numpy as np
import pickle
import sklearn


def preprocessInput(inputData:pd.DataFrame, encoder_path:str, scaler_path:str):
    inputData.drop(labels=["TBG", "referral_source", "TBG_measured", "FTI_measured", "query_on_thyroxine", "query_hyperthyroid", 
                           "query_hypothyroid", "T4U_measured", "TT4_measured", "T3_measured", "TSH_measured"], 
                           axis=1, 
                           inplace=True)
    encoder, scaler = loadTransformers(encoder_path, scaler_path)
    numeric_features = inputData.select_dtypes("number").columns
    categorical_features = inputData.select_dtypes(object).columns

    numericProcessed = pd.DataFrame(data=scaler.transform(inputData[numeric_features]),
                                    columns=numeric_features)

    categoricalPreprocessed = pd.DataFrame(encoder.transform(inputData[categorical_features]).toarray(), 
                             columns = encoder.get_feature_names_out(categorical_features))
    
    final_data = numericProcessed.join(categoricalPreprocessed)
    return final_data

def loadTransformers(encoder_path:str, scaler_path:str):
    encoder = pickle.load(open(encoder_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))
    return encoder, scaler
    
def makePrediction(inputData:pd.DataFrame, encoder_path:str, scaler_path:str, model_path:str):
    processedInput = preprocessInput(inputData, encoder_path, scaler_path)
    model = pickle.load(open(model_path, 'rb'))
    
    result = model.predict(processedInput)

    result_df = pd.DataFrame(data=result, columns=["label"])
    return result_df
 
    