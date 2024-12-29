import os 
import sys
from src.CrditCardDefPred.logger import logging
from src.CrditCardDefPred.exception import CustomException
import pandas as pd
from dotenv import load_dotenv
import pymysql
import pickle
import gzip
import joblib


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import RandomizedSearchCV

load_dotenv()

host=os.getenv("host")
user=os.getenv("user")
passwd=os.getenv("password")
db=os.getenv("db")

def read_sql_data():
    logging.info("Reading sql data")
    try:

        mydb=pymysql.connect(
            host=host,
            user=user,
            password=passwd,
            db=db
        )
        logging.info("connection established")
        df=pd.read_sql_query("SELECT * FROM  uci_credit_card",mydb)
        print (df.head())

        return df
    except Exception as e:
        raise CustomException(e,sys)
    

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with gzip.open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
        # joblib.dump(obj, file_path, compress=3)
        # print(f"Object saved successfully at {file_path}")
    
    except Exception as e:
        raise CustomException(e,sys)
    


def evaluate_models(X_train, y_train, X_test, y_test, models, params):

    try:
        """
        Evaluates multiple models using RandomizedSearchCV and returns their performance.

        Args:
            X_train (np.array): Training features.
            y_train (np.array): Training labels.
            X_test (np.array): Test features.
            y_test (np.array): Test labels.
            models (dict): Dictionary of models to evaluate.
            params (dict): Dictionary of hyperparameters for each model.

        Returns:
            dict: A dictionary containing model names and their test accuracy scores.
        """
        report = {}
        best_params = {}

        for model_name, model in models.items():
            print(f"Training {model_name}...")

            # Fetch hyperparameters for the current model
            param_distributions = params.get(model_name, {})

            # Initialize RandomizedSearchCV
            rs = RandomizedSearchCV(estimator=model,
                                    param_distributions=param_distributions,
                                    n_iter=100,
                                    cv=3,
                                    scoring="accuracy",
                                    verbose=2,
                                    n_jobs=-1,
                                    random_state=42)
            rs.fit(X_train, y_train)

            # Use the best model
            best_model = rs.best_estimator_
            best_params[model_name] = rs.best_params_

            # Predictions
            y_test_pred = best_model.predict(X_test)

            # Calculate test accuracy
            test_accuracy = accuracy_score(y_test, y_test_pred)
            report[model_name] = test_accuracy  # Save test accuracy in the report dictionary

            print(f"{model_name} Test Accuracy: {test_accuracy:.4f}")

        return report,best_params
    
    except Exception as e:
        raise CustomException(e,sys)
        


def load_object(file_path):
    try:
        with gzip.open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
        # Load the object using joblib
        # return joblib.load(file_path)

    except Exception as e:
        raise CustomException(e, sys)