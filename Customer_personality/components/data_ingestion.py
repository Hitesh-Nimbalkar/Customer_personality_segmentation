import os,sys
from six.moves import urllib  
from Customer_personality.constant import *
from Customer_personality.configuration.configuration import Configuration
from Customer_personality.entity.config_entity import DataIngestionConfig
from Customer_personality.entity.artifact_entity import DataIngestionArtifact
from Customer_personality.logger import logging
from Customer_personality.exception import ApplicationException
from Customer_personality.utils.utils import read_yaml_file
from datetime import date
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import shutil



class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            logging.info(f"{'>>'*30}Data Ingestion log started.{'<<'*30} \n\n")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise ApplicationException(e,sys) from e
    

    def download_data(self) -> str:
        try:
            # Raw Data Directory Path
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            
            logging.info(f" Raw Data directory : {raw_data_dir}")

            # Make Raw data Directory
            os.makedirs(raw_data_dir, exist_ok=True)

            # Download URL
            download_url = self.data_ingestion_config.dataset_download_url + "?raw=true"

            # Downloading the zip file
            logging.info(f"Downloading file from URL: {download_url}")
            urllib.request.urlretrieve(download_url, os.path.join(raw_data_dir, "data.zip"))
            logging.info("File downloaded successfully.")

            # Extracting the zip file
            with zipfile.ZipFile(os.path.join(raw_data_dir, "data.zip"), "r") as zip_ref:
                zip_ref.extractall(raw_data_dir)
            logging.info("Zip file extracted successfully.")

            # Delete the downloaded zip file
            os.remove(os.path.join(raw_data_dir, "data.zip"))
            
            # Extracting name of the csv file extracted 
            # Extracted CSV file path (assuming it has a .csv extension)
            csv_file_path = None

            # Get the list of files in the raw data directory
            file_list = os.listdir(raw_data_dir)

            # Search for the CSV file
            for file_name in file_list:
                if file_name.endswith(".csv"):
                    csv_file_path = os.path.join(raw_data_dir, file_name)
                    break
            # Print the name of the CSV file
            if csv_file_path is not None:
                csv_file_name = os.path.basename(csv_file_path)
                logging.info("CSV file name:", csv_file_name)
                
            raw_file_path = os.path.join(raw_data_dir, csv_file_name)
            
            
            # copy the the extracted csv from raw_data_dir ---> ingested Data 
            ingest_file_path=os.path.join(self.data_ingestion_config.ingested_data_dir)
            os.makedirs(ingest_file_path,exist_ok=True)

            
            
            # Copy the extracted CSV file
            shutil.copy2(raw_file_path, ingest_file_path)
            
            # Updating file name 
            # Set the destination directory for ingested data
            ingest_file_path = os.path.join(self.data_ingestion_config.ingested_data_dir, csv_file_name)
            
            
            

            logging.info(f"File: {ingest_file_path} has been downloaded and extracted successfully.")
            
            
            data_ingestion_artifact=DataIngestionArtifact(train_file_path=ingest_file_path,
                                                          is_ingested=True,
                                                          message=f"data ingestion completed successfully"
                                                          )
            logging.info(f"Data Ingestion Artifact:[{data_ingestion_artifact}]")
            return data_ingestion_artifact

        

        except Exception as e:
            raise ApplicationException(e, sys) from e 
        





    def initiate_data_ingestion(self):
        try:
            
            return self.download_data()


        except Exception as e:
            raise ApplicationException(e,sys) from e