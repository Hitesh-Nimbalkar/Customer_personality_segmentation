

# Data Transformation related variables
PIKLE_FOLDER_NAME_KEY='Prediction_Files'
# Artifact
DATA_TRANSFORMATION_ARTIFACT_DIR = "data_transformation"

# key  ---> config.yaml---->values
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
DATA_TRANSFORMATION_DIR_NAME_KEY = "transformed_dir"
DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY = "transformed_train_dir"
DATA_TRANSFORMATION_TEST_DIR_NAME_KEY = "transformed_test_dir"
DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY = "preprocessing_dir"
DATA_TRANSFORMATION_PREPROCESSING_FILE_NAME_KEY = "preprocessed_object_file_name"
DATA_TRANSFORMATION_FEATURE_ENGINEERING_FILE_NAME_KEY ="feature_engineering_object_file_name"


DATA_TRANSFORMATION_INPUT_TRAIN='input_train'
DATA_TRANSFORMATION_TARGET_TRAIN='target_train'
DATA_TRANSFORMATION_INPUT_TEST='input_test'
DATA_TRANSFORMATION_TARGET_TEST='target_test'

import os

# Transformation 
TARGET_COLUMN_KEY='target_column'
NUMERICAL_COLUMN_KEY='numerical_column'
DROP_COLUMNS='drop_columns'

TRANFORMATION_YAML='transformation.yaml'

ROOT_DIR=os.getcwd()
CONFIG_DIR='config'
# Transformation Yaml
TRANSFORMATION_YAML_FILE_PATH=os.path.join(ROOT_DIR,CONFIG_DIR,TRANFORMATION_YAML)


# 

