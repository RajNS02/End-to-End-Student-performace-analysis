import os
import sys
from dataclasses import dataclass

import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import train_models, save_object

@dataclass
class ModelTrainerConfig:
    best_model_file_path = os.path.join("artifacts", "best_model.pkl")

class ModelTrainer:

    def __init__(self):
        self.best_model_config = ModelTrainerConfig()

    def get_best_model(self, models_with_scores):
        try:
            
            test_scores = []
            for model_name in models_with_scores.keys():
                model_score = models_with_scores[model_name]["Test score"]
                test_scores.append((model_score, model_name))
            
            best_score = test_scores[test_scores.index(max(test_scores))][0]
            best_model_name = test_scores[test_scores.index(max(test_scores))][1]
            best_model = models_with_scores[best_model_name]["Model object"]
            
            return best_score, best_model
        
        except Exception as e:
            raise CustomException(e, sys)

    
    def initiate_model_training(self, train_transformed, test_transformed):

        try:
            logging.info("Training start")
            X_train_transformed, y_train, X_test_transformed, y_test = (
                                                                        train_transformed[:, :-1], train_transformed[:, -1],
                                                                        test_transformed[:, :-1], test_transformed[:, -1]
                                                                        )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor()
            }

            params = {
                "Random Forest": {
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features': ['sqrt','log2', None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Decision Tree": {
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter': ['best','random'],
                    # 'max_features': ['sqrt','log2'],
                },
                "Gradient Boosting": {
                    # 'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},

                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate':[.1, .01, 0.5, .001],
                    # 'loss': ['linear','square','exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                "KNeighborsRegressor": {}
            }

            logging.info("Training start!")
            models_with_scores: dict = train_models(models, X_train_transformed, y_train, X_test_transformed, y_test, params, r2_score)
            logging.info("Training has ended! Selecting the best model.")


            best_score, best_model = self.get_best_model(models_with_scores)
            logging.info("Best model selected")

            save_object(
                file_path = self.best_model_config.best_model_file_path,
                obj = best_model
            )
            logging.info("best model saved!")

            print(best_score)
            print(models_with_scores)

        except Exception as e:
            raise CustomException(e, sys)