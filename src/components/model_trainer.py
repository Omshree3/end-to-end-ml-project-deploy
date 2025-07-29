import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException

from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Linear Regression': LinearRegression(),
                'KNeighbors': KNeighborsRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'XGB': XGBRegressor(),#eval_metric='rmse'
                'CatBoost': CatBoostRegressor(verbose=False)
            }
            
            params = {
                'Decision Tree':{
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features': ['auto', 'sqrt', 'log2'],
                    # 'max_depth': [None, 10, 20, 30, 40, 50],
                    # 'min_samples_split': [2, 5, 10],    
                    },
                    
                'Random Forest':{
                    'n_estimators':[8,16,32,64,128]
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features': ['auto', 'sqrt', 'log2'],
                    # 'max_depth': [None, 10, 20, 30, 40, 50],
                    # 'min_samples_split': [2, 5, 10],
                    },
                    
                'Linear Regression':
                    {},
                    
                'KNeighbors': {
                    'n_neighbors': [5, 10, 15, 20],
                    #'weights': ['uniform', 'distance']
                    # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    # 'leaf_size': [30, 40, 50, 60],
                    },
                    
                'AdaBoost':{'n_estimators': [50, 100, 200],
                            'learning_rate': [0.01, 0.1, 1.0]
                            },
                    
                'Gradient Boosting':{
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'n_estimators': [8, 16, 32, 64, 128]
                    # 'max_depth': [3, 5, 7, 9],
                    # 'min_samples_split': [2, 5, 10]
                    },  
                    
                'XGB':{
                    'n_estimators': [8, 16, 32, 64, 128], 
                    'learning_rate': [0.1, 0.01, 0.001]
                    },
                    
                'CatBoost': {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.1, 0.01, 0.001],
                    'iterations': [30, 50, 100]
                }
            }
            
            model_report:dict=evaluate_models(X_train, y_train, X_test, y_test, models, params) ## evaluating the models
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            ## To get best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)  
            
            logging.info("best model found on both training and testing data")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model)
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
        
        
        except Exception as e:
            raise CustomException(e, sys)