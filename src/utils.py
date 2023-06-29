import sys, os
import numpy as np
import pandas as pd
import pickle 
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path) # get directory of folder where we want to save preprocessor object
        os.makedirs(dir_path, exist_ok=True) # makes all missing directories till the dir_path (including dir_path too)

        with open(file_path, "wb") as file_obj: # "wb" -> write in bytes
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

