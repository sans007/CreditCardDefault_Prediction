from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.CrditCardDefPred.pipelines.prediction_pipeline import CustomData,PredictPipeline
from src.CrditCardDefPred.logger import logging
from src.CrditCardDefPred.exception import CustomException
import sys



application = Flask(__name__)
app = application

# Route for the home page
@app.route('/', methods=['GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

# Route for prediction
@app.route('/predictdata', methods=['POST'])
def predict_datapoint():

    try:
     if request.method == 'POST':
            # Gather data from form inputs
            data = CustomData(
                LIMIT_BAL=float(request.form.get('LIMIT_BAL')),
                SEX=int(request.form.get('SEX')),
                MARRIAGE=int(request.form.get('MARRIAGE')),
                EDUCATION=int(request.form.get('EDUCATION')),
                AGE=int(request.form.get('AGE')),
                PAY_0=int(request.form.get('PAY_0')),
                PAY_2=int(request.form.get('PAY_2')),
                PAY_3=int(request.form.get('PAY_3')),
                PAY_4=int(request.form.get('PAY_4')),
                PAY_5=int(request.form.get('PAY_5')),
                PAY_6=int(request.form.get('PAY_6')),
                BILL_AMT1=float(request.form.get('BILL_AMT1')),
                BILL_AMT2=float(request.form.get('BILL_AMT2')),
                BILL_AMT3=float(request.form.get('BILL_AMT3')),
                BILL_AMT4=float(request.form.get('BILL_AMT4')),
                BILL_AMT5=float(request.form.get('BILL_AMT5')),
                BILL_AMT6=float(request.form.get('BILL_AMT6')),
                PAY_AMT1=float(request.form.get('PAY_AMT1')),
                PAY_AMT2=float(request.form.get('PAY_AMT2')),
                PAY_AMT3=float(request.form.get('PAY_AMT3')),
                PAY_AMT4=float(request.form.get('PAY_AMT4')),
                PAY_AMT5=float(request.form.get('PAY_AMT5')),
                PAY_AMT6=float(request.form.get('PAY_AMT6'))
            )

            # Convert input data to DataFrame
            pred_df = data.get_data_as_data_frame()
            print("Input DataFrame:\n", pred_df)
            logging.info("Input DataFrame completed")

            # Load prediction pipeline and make prediction
            predict_pipeline = PredictPipeline()
            print("Starting prediction...")
            logging.info("Prediction started")
            results = predict_pipeline.predict(pred_df)
            print("Prediction Results:", results)
            logging.info("Prediction result in numeric started")

            result = "Have Default next month" if results[0] == 1 else "Have No default next month"
            logging.info("Prediction completed")

            # Pass the result to the template
            return render_template('index.html', results=result)
            
     
    except Exception as e:
        raise CustomException(e,sys)



if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)
