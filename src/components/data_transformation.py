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

from src.utils import save_object

@dataclass
class DataTransformationConfig: # defines where to save the train and test data 
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_features, categorical_features):

        try:
            numerical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "median")), # bcoz there were a lot of outliers, we used median
                    ("standard_scaler", StandardScaler(with_mean = True)) # https://stackoverflow.com/a/57350086
                ]
            )

            logging.info("Numerical features: Transformation done!")

            categorical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "most_frequent")), # Using mode basically
                    ("one_hot_encoder", OneHotEncoder()), # Bcoz for each category, very few categories exist
                    ("standard_scaler", StandardScaler())
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

            numerical_features = [feature for feature in train_df.columns if train_df[feature].dtype != 'O']
            categorical_features = [feature for feature in train_df.columns if train_df[feature].dtype == 'O']
            
            preprocessor = self.get_data_transformer_object(numerical_features, categorical_features)
            target_column = list(train_df.columns())[-1]

            X_train = train_df.drop(columns = [target_column], axis = 1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns = [target_column], axis = 1)
            y_test = test_df[target_column]

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

            return (
                train_df_transformed,
                test_df_transformed
                # self.transformation_config.preprocessor_obj_file_path,
            )
        

        except Exception as e:
            raise CustomException(e,sys)