

#to store the log file
import os
from datetime import datetime

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"

CURRENT_TIME_STAMP = get_current_time_stamp()
ROOT_DIR = os.getcwd()  # to get current working directory

CONFIG_DIR= "config"
CONFIG_FILE_NAME= "config.yaml"
SCHEMA_FILE_NAME='schema.yaml'

CONFIG_FILE_PATH= os.path.join(ROOT_DIR, CONFIG_DIR, CONFIG_FILE_NAME)
SCHEMA_FILE_PATH=os.path.join(ROOT_DIR, CONFIG_DIR, SCHEMA_FILE_NAME)




from Customer_personality.constant.training_pipeline import *
from Customer_personality.constant.training_pipeline.data_ingestion import *
from Customer_personality.constant.training_pipeline.data_validation import *
from Customer_personality.constant.training_pipeline.data_transformation import *
from Customer_personality.constant.training_pipeline.model_trainer import *

SAVED_MODEL_DIRECTORY='Saved_Model'