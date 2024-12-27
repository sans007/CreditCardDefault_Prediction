#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Install required Python dependencies
echo "Installing required dependencies..."
pip install -r requirements.txt

# Generate preprocessor.pkl and model.pkl
echo "Running setup.py to create preprocessor.pkl and model.pkl..."
python pklscript.py

# Start the prediction API service
# echo "Starting the prediction service..."
# python app.py
