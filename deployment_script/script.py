import os
import sys
import pandas as pd
import numpy as np
from src.CrditCardDefPred.logger import logging
from src.CrditCardDefPred.exception import CustomException
from src.CrditCardDefPred.components.data_transformation import DataTransformation
from sklearn.ensemble import RandomForestClassifier
from src.CrditCardDefPred.utils import evaluate_models,save_object
import mlflow
import mlflow.sklearn
import dagshub
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from dataclasses import dataclass


@dataclass
class DataPreprocessingConfig:
     preprocessor_obj_file_path = os.path.join("deployment_script", "preprocessor.pkl")

class DataPreprocessing:
    def __init__(self):
        self.data_preprocessing_config = DataPreprocessingConfig()
    
    def Create_preprocesserFile(self,train_path,test_path):
        try:
                # Load datasets
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)

                logging.info("Reading train and test datasets.")


                # Feature engineering (same for train and test)
                for df in [train_df, test_df]:
                    df['EDUCATION'] = np.where(df['EDUCATION'].isin([0, 5, 6]), 4, df['EDUCATION'])
                    df['MARRIAGE'] = np.where(df['MARRIAGE'] == 0, 3, df['MARRIAGE'])

                    df.drop(columns=['ID'], inplace=True, errors="ignore")

                # # Get preprocessor
                preprocessing_obj = DataTransformation.get_data_transformer_object(self)

                target_column_name = "default_payment_next_month"

                input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
                target_feature_train_df=train_df[target_column_name]
            
                ## divide the test dataset to independent and dependent feature
                input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
                target_feature_test_df=test_df[target_column_name]
                
                logging.info("Applying Preprocessing on training and test dataframe")

                input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

                train_arr = np.c_[
                    input_feature_train_arr, np.array(target_feature_train_df)
                ]
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

                # Save preprocessor object
                save_object(
                    file_path=self.data_preprocessing_config.preprocessor_obj_file_path,
                    obj=preprocessing_obj
                )
                
                return (train_arr,
                        test_arr,
                        self.data_preprocessing_config.preprocessor_obj_file_path
                )
        except Exception as e:
                raise CustomException(e,sys)

@dataclass
class DataModelConfig:
     model_file_path = os.path.join("deployment_script","model.pkl")

class Datamodeling:
    def __init__(self):
          self.data_model_config = DataModelConfig()
    def eval_metrics(self,actual, pred):
            Accuracy_score = accuracy_score(actual, pred)
            Precision_score = precision_score(actual, pred)
            Recall_score = recall_score(actual, pred)
            F1_score = f1_score(actual, pred)
            return Accuracy_score, Precision_score, Recall_score, F1_score
    def Create_modelfile(self,train_arr,test_arr):
        try:
                logging.info("Split training and test input data")
                
                X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
                X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
                # Define models
                models = {
                    "Random Forest": RandomForestClassifier(),
                    
                }

                # Define hyperparameters for RandomizedSearchCV
                params = {
                    "Random Forest": {
                        "max_depth": [5, 8, 15, None],
                        "max_features": [5, 7, "sqrt", 8],
                        "min_samples_split": [2, 8, 15, 20],
                        "n_estimators": [100, 200, 500, 1000]
                    }

                }

                # Evaluate models
                model_report, best_params = evaluate_models(X_train, y_train, X_test, y_test, models, params)

                # Get the best model
                best_model_name = max(model_report, key=model_report.get)  # Get the key with the highest value
                best_model_score = model_report[best_model_name]
                best_model_params = best_params[best_model_name]
                best_model = models[best_model_name]

                print(f"\nBest Model: {best_model_name} with Test Accuracy: {best_model_score:.4f}")
                print(f"Best Parameters for {best_model_name}: {best_model_params}")

                

                
                # Train the best model on the full training set
                best_model.set_params(**best_model_params)
                best_model.fit(X_train, y_train)

                
                # mlflow
                # dagshub.init(repo_owner='sans007', repo_name='CreditCardDefault_Prediction', mlflow=True)
                # mlflow.set_registry_uri("https://dagshub.com/sans007/CreditCardDefault_Prediction.mlflow")
                # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

                # mlflow

                # with mlflow.start_run():

                #     predicted_qualities = best_model.predict(X_test)

                #     # Evaluate the model's performance
                #     Accuracy_score,Precision_score,Recall_score,F1_score = self.eval_metrics(y_test, predicted_qualities)

                #     mlflow.log_params(best_model_params)

                #     mlflow.log_metric("Accuracy", Accuracy_score)
                #     mlflow.log_metric("Precision", Precision_score)
                #     mlflow.log_metric("Recall", Recall_score)
                #     mlflow.log_metric("F1", F1_score)
                    


                #     # Model registry does not work with file store
                #     if tracking_url_type_store != "file":

                #         # Register the model
                #         # There are other ways to use the Model Registry, which depends on the use case,
                #         # please refer to the doc for more information:
                #         # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                #         mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model_name)
                #     else:
                #         mlflow.sklearn.log_model(best_model, "model")


                if best_model_score<0.6:
                    raise CustomException("No best model found")
                logging.info(f"Best found model on both training and testing dataset")

                save_object(
                    file_path=self.data_model_config.model_file_path,
                    obj=best_model
                )

            

        
        except Exception as e:
            raise CustomException(e,sys)

    
    