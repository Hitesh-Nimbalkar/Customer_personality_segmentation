from collections import namedtuple

#data Ingestion

DataIngestionArtifact = namedtuple('DataIngestionArtifact',
                                   ['train_file_path','is_ingested','message'])



DataValidationArtifact = namedtuple("DataValidationArtifact",
["schema_file_path","is_validated","message","validated_train_path"])

DataTransformationArtifact = namedtuple("DataTransformationArtifact",["is_transformed",
                                                                    "message","Feature_eng_train_file_path",
                                                                    "transformed_train_file_path",
                                                                    "preprocessed_object_file_path",
                                                        "feature_engineering_object_file_path"])

ModelTrainerArtifact =namedtuple("ModelTrainerArtifact",[
                                                            "is_trained",
                                                            "message",
                                                            "model_selected",
                                                            "model_prediction_png",
                                                            "model_name",
                                                            "report_path",
                                                            "csv_file_path"

                                                        ])

ModelEvaluationArtifact=namedtuple("ModelEvaluationArtifact",["model_name",
                                                              "Silhouette_score",
                                                              "selected_model_path","model_prediction_png","optimal_cluster","rfm_csv_path",
                                                              "model_report_path"])

ModelPusherArtifact=namedtuple("ModelPusherArtifact",["message"])