## End to End Credit Card Default Prediction Data Science Project

### Create Environment
virtualenv .sanenv
### activate
source .sanenv/Scripts/activate
### deactivate
deactivate

### install requirements tools
pip install -r requirements.txt

### Project structure
setup.py
template.py

### Development
src/CREDITCARDefPred/components,pipelines,logger.py,exceptions.py,utils.py

### Dataset structure
DataIngestion
 
### For feature engineering
DataTransformation

### For Model Training
ModelTraining

### For Prediction
PredictionPipeline

### For API
app.py