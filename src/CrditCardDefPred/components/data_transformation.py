import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.CrditCardDefPred.logger import logging
from src.CrditCardDefPred.exception import CustomException
from src.CrditCardDefPred.utils import save_object
import os



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self,df):
        """
        This function constructs a preprocessing object with pipelines for both
        categorical and numerical data transformations.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            preprocessor (ColumnTransformer): The preprocessing pipeline object.
        
        """
        try:
            # Identify categorical and numerical columns
            enco_columns = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
            unenco_columns = [col for col in df.columns if col not in enco_columns and col != 'default_payment_next_month']

            # Pipelines for numerical and categorical columns
            unenco_columns_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ('scalar', StandardScaler())
            ])

            enco_columns_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(sparse_output=False)),
                ("scaler", StandardScaler(with_mean=False))
            ])

            # Create ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("unenco_columns_pipeline", unenco_columns_pipeline, unenco_columns),
                    ("enco_columns_pipeline", enco_columns_pipeline, enco_columns)
                ]
            )
            
            return preprocessor

        except Exception as e:
            raise Exception(e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads, preprocesses, and balances datasets, then applies transformations.

        Args:
            train_path (str): Path to training data.
            test_path (str): Path to test data.

        Returns:
            tuple: Preprocessed training and test arrays and preprocessor object path.
        """
        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test datasets.")


            # Feature engineering (same for train and test)
            for df in [train_df, test_df]:
                df['EDUCATION'] = np.where(df['EDUCATION'].isin([0, 5, 6]), 4, df['EDUCATION'])
                df['MARRIAGE'] = np.where(df['MARRIAGE'] == 0, 3, df['MARRIAGE'])

                df.replace({
                    'SEX': {1: 'MALE', 2: 'FEMALE'},
                    'EDUCATION': {1: 'graduate school', 2: 'university', 3: 'high school', 4: 'others'},
                    'MARRIAGE': {1: 'married', 2: 'single', 3: 'others'}
                }, inplace=True)

                df.drop(columns=['ID'], inplace=True, errors="ignore")

            # # Get preprocessor
            preprocessing_obj = self.get_data_transformer_object(train_df)

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
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(sys,e)