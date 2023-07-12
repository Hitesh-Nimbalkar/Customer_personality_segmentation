from collections import namedtuple


DataIngestionConfig= namedtuple("DataIngestionConfig",
                                ["dataset_download_url",
                                 "ingested_data_dir",
                                 "raw_data_dir",
                                 "ingested_train_dir",
                                 ])



TrainingPipelineConfig= namedtuple("TrainingPipelineConfig",["artifact_dir"])

DataValidationConfig = namedtuple("DataValidationConfig", ["schema_file_path","validated_train_path"])

DataTransformationConfig = namedtuple("DataTransformationConfig",["transformed_train_dir",
                                                                  "preprocessed_object_file_path",
                                                                  "feature_engineering_object_file_path"])
ModelTrainerConfig = namedtuple("ModelTrainerConfig",["trained_model_directory",
                                                      "trained_model_file_path",'png_location',
                                                      "model_config_path","report_path"])

SavedModelConfig = namedtuple("SavedModelConfig",["saved_model_file_path","saved_model_csv",
                                            "saved_report_file_path","saved_model_prediction_png"])
                                    



