import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl') ## Path to save the preprocessor object
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig() ## Initialize the configuration for data transformation
        
    def get_data_transformer_object(self): ## Method to create a data transformation pipeline
        ''' This method creates a data transformation pipeline that includes preprocessing steps for numerical and categorical features.
            It returns a ColumnTransformer object that applies different transformations to specified columns.'''
        
        try:
            numerical_columns =["writing_score", "reading_score"]
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),  # Impute missing values with median
                ('scaler', StandardScaler()) # Scale numerical features
                ]
            )
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),# Impute missing
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')), # One-hot encode categorical features
                ('scaler', StandardScaler(with_mean=False)) # Scale categorical features                     
            
                ]
            )
            
            logging.info(f'categorical columns: {categorical_columns}')
            logging.info(f'numerical columns: {numerical_columns}')
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipelines', cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path): ## Method to initiate data transformation
        
        try:
            train_df = pd.read_csv(train_path)  # Load training data
            test_df = pd.read_csv(test_path)    # Load testing data
            
            logging.info('Read train and test data completed')
            
            logging.info("obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object()  # Get the preprocessor object
            
            target_column_name = 'math_score'  # Specify the target column
            numerical_columns = ["writing_score", "reading_score"]
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)  # Drop target column from training data
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)  # Drop target column from testing data
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]  # Combine features and target for training data
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]  # Combine features and target for test data
            
            logging.info(f"Saved preprocessing object")
            
            save_object (
                
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj  # Save the preprocessor object to the specified file path
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )  # Return transformed data and preprocessor file path
            
        except Exception as e:
            raise CustomException(e, sys)