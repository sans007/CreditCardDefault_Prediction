from src.CrditCardDefPred.logger import logging
from src.CrditCardDefPred.exception import CustomException
import sys

if __name__ == '__main__':
    logging.info("the execution has started")


    try:

        a=1/10

    except Exception as e:
        logging.info("Custom exception")
        raise CustomException(e,sys)
