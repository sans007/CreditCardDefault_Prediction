from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.CrditCardDefPred.pipelines.prediction_pipeline import CustomData,PredictPipeline


application = Flask(__name__)
main_app = application

# Route for the home page
@main_app.route('/', methods=['GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

# Route for prediction
@main_app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    if request.method == 'POST':
            # Gather data from form inputs
            data = CustomData(
                LIMIT_BAL= 120000,
                SEX=2,
                EDUCATION=2,
                MARRIAGE=2,
                AGE=26,
                PAY_0=-1,
                PAY_2=2,
                PAY_3=0,
                PAY_4=0,
                PAY_5=0,
                PAY_6=2,
                BILL_AMT1=2682,
                BILL_AMT2=3272,
                BILL_AMT3=3455,
                BILL_AMT4=3261,
                BILL_AMT5=3455,
                BILL_AMT6=3261,
                PAY_AMT1=0,
                PAY_AMT2=1000,
                PAY_AMT3=1000,
                PAY_AMT4=1000,
                PAY_AMT5=0,
                PAY_AMT6=2000
            )
            # data = CustomData(
            #     LIMIT_BAL=float(request.form.get('LIMIT_BAL')),
            #     SEX=int(request.form.get('SEX')),
            #     MARRIAGE=int(request.form.get('MARRIAGE')),
            #     EDUCATION=int(request.form.get('EDUCATION')),
            #     AGE=int(request.form.get('AGE')),
            #     PAY_0=int(request.form.get('PAY_0')),
            #     PAY_2=int(request.form.get('PAY_2')),
            #     PAY_3=int(request.form.get('PAY_3')),
            #     PAY_4=int(request.form.get('PAY_4')),
            #     PAY_5=int(request.form.get('PAY_5')),
            #     PAY_6=int(request.form.get('PAY_6')),
            #     BILL_AMT1=float(request.form.get('BILL_AMT1')),
            #     BILL_AMT2=float(request.form.get('BILL_AMT2')),
            #     BILL_AMT3=float(request.form.get('BILL_AMT3')),
            #     BILL_AMT4=float(request.form.get('BILL_AMT4')),
            #     BILL_AMT5=float(request.form.get('BILL_AMT5')),
            #     BILL_AMT6=float(request.form.get('BILL_AMT6')),
            #     PAY_AMT1=float(request.form.get('PAY_AMT1')),
            #     PAY_AMT2=float(request.form.get('PAY_AMT2')),
            #     PAY_AMT3=float(request.form.get('PAY_AMT3')),
            #     PAY_AMT4=float(request.form.get('PAY_AMT4')),
            #     PAY_AMT5=float(request.form.get('PAY_AMT5')),
            #     PAY_AMT6=float(request.form.get('PAY_AMT6'))
            # )

            # Convert input data to DataFrame
            pred_df = data.get_data_as_data_frame()
            print("Input DataFrame:\n", pred_df)

            # Load prediction pipeline and make prediction
            predict_pipeline = PredictPipeline()
            print("Starting prediction...")
            results = predict_pipeline.predict(pred_df)
            print("Prediction Results:", results)

            # Render the home page with the prediction result
            return render_template('index.html', results=results[0])



if __name__ == "__main__":
    # Run the Flask app
    main_app.run(host="0.0.0.0", port=5000, debug=True)
