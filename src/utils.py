import sys, os, json

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

import pickle 
from src.exception import CustomException


def save_features(all_features, numerical_features, categorical_features, file_path):
    features = {}
    for feat in all_features:
        if feat in numerical_features:
            features[feat] = "numerical_feature"
        elif feat in categorical_features:
            features[feat] = "categorical_feature"

    with open(file_path, "w") as file:
        json.dump(features, file)


def save_cat_var_options(categorical_features_with_options, file_path):
    with open(file_path, "w") as file:
        json.dump(categorical_features_with_options, file)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path) # get directory of folder where we want to save preprocessor object
        os.makedirs(dir_path, exist_ok=True) # makes all missing directories till the dir_path (including dir_path too)

        with open(file_path, "wb") as file_obj: # "wb" -> write in bytes
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def load_saved_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        
        return obj

    except Exception as e:
        raise CustomException(e, sys)
    

def train_models(models, X_train_transformed, y_train, X_test_transformed, y_test, params, metric):
    try:
        models_with_scores = {}

        for model_name in models.keys():

            model = models[model_name]
            model_params = params[model_name]

            gs = GridSearchCV(model, model_params, cv = 3)
            gs.fit(X_train_transformed, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train_transformed, y_train)

            y_train_predict = model.predict(X_train_transformed)
            y_test_predict = model.predict(X_test_transformed)
            
            models_with_scores[model_name] = {}
            models_with_scores[model_name]["Model object"] = model
            # models_with_scores[model_name]["Training score"] = {}
            # models_with_scores[model_name]["Test score"] = {}

            models_with_scores[model_name]["Training score"] = metric(y_train, y_train_predict) 
            models_with_scores[model_name]["Test score"] = metric(y_test, y_test_predict) 

        return models_with_scores
    
    except Exception as e:
        raise CustomException(e, sys)