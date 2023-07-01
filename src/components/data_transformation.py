import sys, os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, save_features, save_cat_var_options

@dataclass
class DataTransformationConfig: # defines where to save the preprocessor object
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_features, categorical_features):

        try:
            numerical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "median")), # bcoz there were a lot of outliers, we used median
                    ("standard_scaler", StandardScaler(with_mean = False)) # https://stackoverflow.com/a/57350086
                ]
            )

            logging.info("Numerical features: Transformation done!")

            categorical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "most_frequent")), # Using mode basically
                    ("one_hot_encoder", OneHotEncoder()), # Bcoz for each category, very few categories exist
                    ("standard_scaler", StandardScaler(with_mean = False))
                ]
            )
            logging.info("Categorical  features: Transformation done!")

            preprocessor = ColumnTransformer(
                [("numerical_pipeline", numerical_pipeline, numerical_features),
                 ("categorical_pipeline", categorical_pipeline, categorical_features),
                ]
            )

            logging.info("Transformation done!")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, training_data_path, test_data_path):

        try: 
            train_df = pd.read_csv(training_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Reading training and test data complete!")
    
            target_column = list(train_df.columns)[-1]
            y_train = train_df[target_column]
            X_train = train_df.drop(columns = [target_column], axis = 1)
            y_test = test_df[target_column]
            X_test = test_df.drop(columns = [target_column], axis = 1)

            numerical_features = [feature for feature in X_train.columns if X_train[feature].dtype != 'O']
            categorical_features = [feature for feature in X_train.columns if X_train[feature].dtype == 'O']
            
            # print(numerical_features) -> working
            preprocessor = self.get_data_transformer_object(numerical_features, categorical_features)

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            train_df_transformed = np.c_[
                X_train_transformed, np.array(y_train)
            ]
            test_df_transformed = np.c_[X_test_transformed, np.array(y_test)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path = self.transformation_config.preprocessor_obj_file_path,
                obj = preprocessor
            )

            logging.info("Saving numerical and categorical features")
            categorical_features_with_options = {}
            for cat_feat in categorical_features:
                categorical_features_with_options[cat_feat] = list(X_train[cat_feat].unique())
            
            save_features(list(X_train.columns), numerical_features, categorical_features, "artifacts\Features.json")
            save_cat_var_options(categorical_features_with_options, "artifacts\cat_var_options.json")

            return (
                train_df_transformed,
                test_df_transformed
                # self.transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)