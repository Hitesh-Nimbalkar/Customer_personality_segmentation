
from Customer_personality.exception import ApplicationException
from Customer_personality.logger import logging
import sys
import os
from Customer_personality.entity.config_entity import *
from Customer_personality.entity.artifact_entity import *
from Customer_personality.constant import *
from Customer_personality.configuration.configuration import Configuration
from Customer_personality.utils.utils import read_yaml_file,load_object
from Customer_personality.constant.training_pipeline import *

class ModelEvaluation:


    def __init__(self,
                    data_validation_artifact:DataValidationArtifact,
                    model_trainer_artifact:ModelTrainerArtifact):
        
        try:

            self.data_validation_artifact=data_validation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            
            # Saved Model config 
            self.config=Configuration()
            self.saved_model_config=self.config.saved_model_config()
            
        except Exception as e:
            raise ApplicationException(e,sys)
        
        
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info(" Model Evaluation Started ")
            ## Artifact trained Model  files
            model_trained_artifact_path = self.model_trainer_artifact.model_selected
            model_trained_report = self.model_trainer_artifact.report_path
            artifact_model_prediction_png=self.model_trainer_artifact.model_prediction_png
            artifact_rfm_table=self.model_trainer_artifact.csv_file_path
            
            # Saved Model files

            saved_model_path = self.saved_model_config.saved_model_file_path
            saved_model_report_path=self.saved_model_config.saved_report_file_path
            saved_model_prediction_png=self.saved_model_config.saved_model_prediction_png
            saved_model_rfm_table=self.saved_model_config.saved_model_csv

                        
            logging.info(f" Artifact Trained model : {model_trained_artifact_path}")

            # Load the model evaluation report from the saved YAML file


            # Loading the models
            logging.info("Saved_models directory .....")
            os.makedirs(SAVED_MODEL_DIRECTORY,exist_ok=True)
            
            # Check if SAVED_MODEL_DIRECTORY is empty
            if not os.listdir(SAVED_MODEL_DIRECTORY):
                model_trained_report_data = read_yaml_file(file_path=model_trained_report)
                
                artifact_model_Silhouette_score =float( model_trained_report_data['Silhouette_score'])
                model_name = model_trained_report_data['Model_name']
                Silhouette_score = artifact_model_Silhouette_score
                cluster=int(model_trained_report_data['number_of_clusters'])
                
                # Artifact ----> Model, Model Report 
                model_path = model_trained_artifact_path
                model_report_path = model_trained_report
                png_path=artifact_model_prediction_png
                cluster=cluster
                rfm_csv_path=artifact_rfm_table
                
            else:
                saved_model_report_data = read_yaml_file(file_path=saved_model_report_path)
                model_trained_report_data = read_yaml_file(file_path=model_trained_report)
                
                
                saved_model = load_object(file_path=saved_model_path)
                artifact_model = load_object(file_path=model_trained_artifact_path)

                # Compare the Silhouette_scores 
                saved_model_Silhouette_score = float(saved_model_report_data['Silhouette_score'])

                artifact_model_Silhouette_score =float(model_trained_report_data['Silhouette_score'])

                # Compare the models and log the result
                if artifact_model_Silhouette_score > saved_model_Silhouette_score:
                    logging.info("Trained model outperforms the saved model!")
                    model_path = model_trained_artifact_path
                    model_report_path = model_trained_report
                    png_path=artifact_model_prediction_png
                    model_name = model_trained_report_data['Model_name']
                    Silhouette_score = float( model_trained_report_data['Silhouette_score'])
                    cluster=int(model_trained_report_data['number_of_clusters'])
                    rfm_csv_path=artifact_rfm_table
                    
                    logging.info(f"Model Selected : {model_name}")
                    logging.info(f"Silhouette_score : {Silhouette_score}")
                  
                elif artifact_model_Silhouette_score < saved_model_Silhouette_score:
                    logging.info("Saved model outperforms the trained model!")
                    model_path = saved_model_path
                    model_report_path = saved_model_report_path
                    png_path=saved_model_prediction_png
                    model_name = saved_model_report_data['Model_name']
                    Silhouette_score = float( saved_model_report_data['Silhouette_score'])
                    cluster=int(saved_model_report_data['number_of_clusters'])
                    
                    rfm_csv_path=saved_model_rfm_table
                    logging.info(f"Model Seelcted : {model_name}")

                    logging.info(f"Silhouette_score : {Silhouette_score}")
            
                else:
                    logging.info("Both models have the same Silhouette_score.")
                    model_path = saved_model_path
                    model_report_path = saved_model_report_path
                    png_path=saved_model_prediction_png
                    cluster=int(saved_model_report_data['number_of_clusters'])
                    
                    
                    model_name = saved_model_report_data['Model_name']
                    rfm_csv_path=saved_model_rfm_table
                    
                    Silhouette_score = float( saved_model_report_data['Silhouette_score'])
                    logging.info(f"Model Selected : {model_name}")

                    logging.info(f"Silhouette_score : {Silhouette_score}")
                
                

            # Create a model evaluation artifact
            model_evaluation = ModelEvaluationArtifact(model_name=model_name, 
                                                       Silhouette_score=Silhouette_score,
                                                    selected_model_path=model_path, 
                                                    model_report_path=model_report_path,optimal_cluster=cluster,
                                                    model_prediction_png=png_path,
                                                    rfm_csv_path=rfm_csv_path)

            logging.info("Model evaluation completed successfully!")

            return model_evaluation
        except Exception as e:
            logging.error("Error occurred during model evaluation!")
            raise ApplicationException(e, sys) from e


    def __del__(self):
        logging.info(f"\n{'*'*20} Model evaluation log completed {'*'*20}\n\n")