from src.CrditCardDefPred.logger import logging
from src.CrditCardDefPred.exception import CustomException
import sys
from src.CrditCardDefPred.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.CrditCardDefPred.components.data_transformation import DataTransformation, DataTransformationConfig 
from src.CrditCardDefPred.components.model_trainer import ModelTrainer,ModelTrainerConfig

if __name__ == '__main__':
    logging.info("the execution has started")


    try:
        
        #data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        #data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        #model trainer
        model_trainer=ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))

    except Exception as e:
        logging.info("Custom exception")
        raise CustomException(e,sys)
