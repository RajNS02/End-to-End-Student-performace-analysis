import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

@dataclass # Use when we want to define only variables. With this, we can do without all that __init__ bs
class DataIngestionConfig: # defines where to save the train and test data 
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self, dataset: str):
        self.ingestion_config = DataIngestionConfig()
        self.dataset = dataset
    
    def intitiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df = pd.read_csv("notebook\\data\\" + f"{self.dataset}.csv")
            logging.info("Read the dataset using pandas")

            # Makes artifacts folder
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)
            # Saves df in the path artifacts/data.csv
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            logging.info("Train test split intiated")
            train, test = train_test_split(df, test_size = 0.2, random_state = 42)
            train.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        
# if __name__ == "__main__": # By default, __name__ is __main__. So, whenever we run this particular script, the code in it runs.