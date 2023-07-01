import os, sys

import numpy as np

from src.exception import CustomException
from src.utils import load_saved_object

# def get_array_representation(data):
#     # if type(data).__name__ == "list":
#     #     return np.array(data).reshape(1, -1)
#     if type(data).__name__ == "DataFrame":
#         data = data.values
#         if data.ndim == 1:
#             return data.reshape(1, -1)
#         elif data.ndim == 2:
#             return data
    
def predict(features):
    preprocessor = load_saved_object(os.path.join("artifacts", "preprocessor.pkl"))
    transformed_features = preprocessor.transform(features)

    if transformed_features.ndim == 1:
        transformed_features = transformed_features.reshape(1, -1)
    elif transformed_features.ndim >= 2:
        pass

    model = load_saved_object(os.path.join("artifacts", "best_model.pkl"))
    preds = model.predict(transformed_features)

    return preds

