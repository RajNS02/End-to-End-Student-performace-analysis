from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Start ingesting data and split it into training and test set
obj = DataIngestion("student_info")
train_data_path, test_data_path = obj.intitiate_data_ingestion()

# Transform the training and test set features
data_transformation = DataTransformation()
train_df_transformed, test_df_transformed = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

# Start training:
ModelTrainer().initiate_model_training(train_df_transformed, test_df_transformed)