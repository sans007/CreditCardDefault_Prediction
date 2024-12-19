import os
import sys
from src.CrditCardDefPred.exception import CustomException
from src.CrditCardDefPred.logger import logging
import pandas as pd
from src.CrditCardDefPred.utils import read_sql_data

from sklearn.model_selection import train_test_split

from dataclasses import dataclass

from imblearn.over_sampling import SMOTE

@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts", "raw.csv")
    train_data_path:str=os.path.join("artifacts", "train.csv")
    test_data_path:str=os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            #reading the sql data
            df=read_sql_data()
            logging.info("Reading Completed from mysql database")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            # Validate the presence of target column
            target_column = "default_payment_next_month"
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in the dataset.")

            # Separate features and target variable
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Apply SMOTE for balancing the dataset
            logging.info("Applying SMOTE to balance the dataset...")
            smote = SMOTE(random_state=42)
            x_smote, y_smote = smote.fit_resample(X, y)
            logging.info("SMOTE applied successfully.")

            # Combine balanced features and target into a DataFrame
            df_bal = pd.DataFrame(x_smote, columns=X.columns)
            df_bal[target_column] = y_smote


            train_set,test_set = train_test_split(df_bal,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("Data Ingestion is completed")


            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        

        except Exception as e:
            raise CustomException(e,sys)