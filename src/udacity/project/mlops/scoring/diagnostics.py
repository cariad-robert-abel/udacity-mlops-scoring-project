
import pandas as pd
import numpy as np
import timeit
import os
import json

################## Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 

################## Function to get model predictions
def model_predictions():
    # Read the deployed model and a test dataset, calculate predictions
    return # Return value should be a list containing all predictions

################## Function to get summary statistics
def dataframe_summary():
    # Calculate summary statistics here
    return # Return value should be a list containing all summary statistics

##################Function to get timings
def execution_time():
    # Calculate timing of training.py and ingestion.py
    return # Return a list of 2 timing values in seconds

################## Function to check dependencies
def outdated_packages_list():
    # Get a list of 


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
