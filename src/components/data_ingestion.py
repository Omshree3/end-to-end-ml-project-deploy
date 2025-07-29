import os ## to handle file paths
import sys ## to handle system-specific parameters and functions
from src.exception import CustomException ## to handle custom exceptions
from src.logger import logging ## to log messages
from sklearn.model_selection import train_test_split ## to split the dataset into training and testing sets
import pandas as pd
from dataclasses import dataclass ## to create classes for data storage
from src.components.data_transformation import DataTransformation ## to use the data transformation component
from src.components.data_transformation import DataTransformationConfig ## to use the configuration for data transformation

from src.components.model_trainer import ModelTrainerConfig ## to use the configuration for model training
from src.components.model_trainer import ModelTrainer ## to use the model trainer component

@dataclass
class DataIngestionconfig:
    train_data_path:str =os.path.join('artifacts','train.csv')
    test_data_path:str =os.path.join('artifacts','test.csv')
    raw_data_path:str =os.path.join('artifacts','raw.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig() ## initializing the configuration for data ingestion
        
    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            df=pd.read_csv('notebook\data\stud.csv') ## reading the dataset from a CSV file
            logging.info('Read the dataset as a dataframe')
            
            ##artifact folder creation
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) ## saving the raw data
            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42) ## splitting the data into training and testing sets
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True) ## saving the training set       
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) ## saving the testing set
            
            logging.info('Ingestion of the data is completed')
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path ) ## returning the paths of the train, test  files
            
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion() ## running the data ingestion process
    
    data_transformation = DataTransformation() ## creating an instance of the DataTransformation class
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data, test_data) ## initiating the data transformation process
    
    modeltrainer = ModelTrainer() ## creating an instance of the ModelTrainer class
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr)) ## initiating the model training process
    