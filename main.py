from src.CrditCardDefPred.logger import logging
from src.CrditCardDefPred.exception import CustomException
import sys
from src.CrditCardDefPred.components.data_ingestion import DataIngestion, DataIngestionConfig

if __name__ == '__main__':
    logging.info("the execution has started")


    try:
        
        #data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()

    except Exception as e:
        logging.info("Custom exception")
        raise CustomException(e,sys)
