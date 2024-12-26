from src.CrditCardDefPred.logger import logging
from src.CrditCardDefPred.exception import CustomException
import sys
import pandas as pd
from deployment_script.script import DataPreprocessing,DataPreprocessingConfig,Datamodeling,DataModelConfig
from src.CrditCardDefPred.components.data_ingestion import DataIngestion

if __name__ == '__main__':
    logging.info("the execution has started")


    try:
        data_ingestion = DataIngestion()
        train_path,test_path=data_ingestion.initiate_data_ingestion()

        data_preprossing = DataPreprocessing()
        train_arr,test_arr,_=data_preprossing.Create_preprocesserFile(train_path,test_path)

        
        modeling=Datamodeling()
        print(modeling.Create_modelfile(train_arr,test_arr))

    except Exception as e:
        logging.info("Custom exception")
        raise CustomException(e,sys)
