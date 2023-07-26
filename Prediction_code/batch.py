import os
import logging
from Customer_personality.logger import logging
from Customer_personality.exception import ApplicationException
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from Customer_personality.utils.utils import read_yaml_file,load_object
from Customer_personality.constant import *
from Customer_personality.constant import *
from sklearn.decomposition import PCA
import urllib
import yaml
import numpy as np
import os
import seaborn as sns 
import itertools
import matplotlib.pyplot as plt

PREDICTION_FOLDER='batch_Prediction'
PREDICTION_CSV='prediction_csv'
PREDICTION_FILE='prediction.csv'

FEATURE_ENG_FOLDER='feature_eng'

ROOT_DIR=os.getcwd()
FEATURE_ENG=os.path.join(ROOT_DIR,PREDICTION_FOLDER,FEATURE_ENG_FOLDER)
BATCH_PREDICTION=os.path.join(ROOT_DIR,PREDICTION_FOLDER,PREDICTION_CSV)
BATCH_COLLECTION_PATH ='batch_prediction'




class batch_prediction:
    def __init__(self,input_file_path, 
                 model_file_path, 
                 transformer_file_path, 
                 feature_engineering_file_path) -> None:
        
        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.transformer_file_path = transformer_file_path
        self.feature_engineering_file_path = feature_engineering_file_path

        
    
    def start_batch_prediction(self):
    
            logging.info("Loading the saved pipeline")

            # Load the feature engineering pipeline
            with open(self.feature_engineering_file_path, 'rb') as f:
                feature_pipeline = pickle.load(f)

            logging.info(f"Feature eng Object acessed :{self.feature_engineering_file_path}")
            
            
            # Load the data transformation pipeline
            with open(self.transformer_file_path, 'rb') as f:
                preprocessor = pickle.load(f)

            logging.info(f"Preprocessor  Object accessed :{self.transformer_file_path}")
            
            # Load the model separately
            model =load_object(file_path=self.model_file_path)

            logging.info(f"Model File Path: {self.model_file_path}")


            # Create the feature engineering pipeline
            feature_engineering_pipeline = Pipeline([
                ('feature_engineering', feature_pipeline)
            ])

            # Read the input file
            df = pd.read_csv(self.input_file_path,sep='\t')
        
            # Apply feature engineering
            df = feature_engineering_pipeline.transform(df)

            # Save the feature-engineered data as a CSV file
            FEATURE_ENG_PATH = FEATURE_ENG  # Specify the desired path for saving the file
            os.makedirs(FEATURE_ENG_PATH, exist_ok=True)
            file_path = os.path.join(FEATURE_ENG_PATH, 'batch_fea_eng.csv')
            df.to_csv(file_path, index=False)
            logging.info("Feature-engineered batch data saved as CSV.")
            
            feature_df=df.copy()
            # Encoding 
                        
            # Ecoding categorical features
            education_mapping = {
                'Graduate': 3,
                'Postgraduate': 2,
                'Undergraduate': 1
            }

            feature_df['Education'] = feature_df['Education'].map(education_mapping).astype('int')
            
            feature_df['Living_With'] = feature_df['Living_With'].map({'Alone': 0, 'Partner': 1}).astype(int)
            
          
            logging.info(f"Columns before transformation: {', '.join(f'{col}: {feature_df[col].dtype}' for col in feature_df.columns)}")
            
            # Transform the feature-engineered data using the preprocessor
            transformed_data = preprocessor.transform(feature_df)
            logging.info(f"Transformed Data Shape: {transformed_data.shape}")
            
            logging.info("Transformation completed successfully")
            col =feature_df.columns
            transformed_train_df = pd.DataFrame(transformed_data, columns=col )
            
            # Saving preprocessed dataframe 
            file_path = os.path.join(FEATURE_ENG_PATH, 'preprocessed.csv')
            transformed_train_df.to_csv(file_path, index=False)
            
            # PCA Transformatio 
            logging.info(" PCA Transformation Started ........")
            # Setting n_components=3
            pca = PCA(n_components=3)  # Create an instance of the PCA class
            pca.fit(transformed_train_df)  # Fit the PCA model to the data
            transformed_data = pca.transform(transformed_train_df) 
            
            df_pca = pd.DataFrame(transformed_data, columns=['F1', 'F2','F3'])
            df_pca = df_pca.loc[:, ~df_pca.columns.str.contains('^Unnamed')]
            
            # Saving preprocessed dataframe 
            file_path = os.path.join(FEATURE_ENG_PATH, 'pca.csv')
            df_pca.to_csv(file_path)
            
            predictions = model.predict(df_pca)
            logging.info(f"Predictions done :{predictions}")
            
            
            # Create a DataFrame from the predictions array
            df_predictions = pd.DataFrame(predictions, columns=['cluster'])
            
            
            #Adding cluster labels to the Dataframe
            feature_df['cluster']=df_predictions['cluster']
            
            
         #   feature_df.to_csv('Prediction.csv')

    
            # Save the predictions to a CSV file
            BATCH_PREDICTION_PATH = BATCH_PREDICTION  # Specify the desired path for saving the CSV file
            os.makedirs(BATCH_PREDICTION_PATH, exist_ok=True)
            csv_path = os.path.join(BATCH_PREDICTION_PATH,'predictions.csv')
            
            feature_df.to_csv(csv_path)
            logging.info(f"Batch predictions saved to '{csv_path}'.")
            
            
            return csv_path
    
