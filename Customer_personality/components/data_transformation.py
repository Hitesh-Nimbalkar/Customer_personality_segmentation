import os 
import sys
import pandas as pd
import numpy as np
from Customer_personality.logger import logging
from Customer_personality.exception import ApplicationException
from Customer_personality.entity.artifact_entity import *
from Customer_personality.entity.config_entity import *
from Customer_personality.utils.utils import read_yaml_file,save_data,save_object
from Customer_personality.constant import *

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler,LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler

class Feature_Engineering(BaseEstimator, TransformerMixin):
    
    def __init__(self,drop_columns):
        
        """
        This class applies necessary Feature Engneering 
        """
        logging.info(f"\n{'*'*20} Feature Engneering Started {'*'*20}\n\n")
        

                                ############### Accesssing Column Labels #########################
                                
                                
                 #   Schema.yaml -----> Data Tranformation ----> Method: Feat Eng Pipeline ---> Class : Feature Eng Pipeline              #
                                
                                

        self.columns_to_drop = drop_columns

        
                                ########################################################################
        
        logging.info(f" Numerical Columns , Categorical Columns , Target Column initialised in Feature engineering Pipeline ")


    # Feature Engineering Pipeline 
    
    
    
    
                                                    ######################### Data Modification ############################

    def drop_columns(self,X:pd.DataFrame):
        try:
            columns=X.columns
            
            logging.info(f"Columns before drop  {columns}")
            
            # Columns Dropping
            drop_column_labels=self.columns_to_drop
            
            logging.info(f" Dropping Columns {drop_column_labels} ")
            
            X=X.drop(columns=drop_column_labels,axis=1)
            
            return X
        
        except Exception as e:
            raise ApplicationException(e,sys) from e
    def drop_rows_with_nan(self, X: pd.DataFrame):
        # Log the shape before dropping NaN values
        logging.info(f"Shape before dropping NaN values: {X.shape}")
        
        # Drop rows with NaN values
        X = X.dropna()
        #X.to_csv("Nan_values_removed.csv", index=False)
        
        # Log the shape after dropping NaN values
        logging.info(f"Shape after dropping NaN values: {X.shape}")
        
        logging.info("Dropped NaN values.")
        
        return X
 
    def drop_duplicates(self,X:pd.DataFrame):
        """
        Drops duplicate rows from a pandas DataFrame and returns the modified DataFrame.
        
        Args:
            df (pandas.DataFrame): The DataFrame to remove duplicate rows from.
            
        Returns:
            pandas.DataFrame: The modified DataFrame with duplicate rows removed.
        """
        
        print(" Drop duplicate value")
        X = X.drop_duplicates()
        
        
        return X
   


    def remove_duplicate_rows_keep_last(self,X):
        
        logging.info(f"DataFrame shape before removing duplicates: {X.shape}")
        num_before = len(X)
        X.drop_duplicates(inplace = True)
        num_after = len(X)
        
        num_duplicates = num_before - num_after
        logging.info(f"Removed {num_duplicates} duplicate rows")
        logging.info(f"DataFrame shape after removing duplicates: {X.shape}")
        
        return X


    def convert_nan_null_to_nan(self,X:pd.DataFrame):
        # Convert "NAN" and "NULL" values to np.nan
        X.replace(["NAN", "NULL","nan"], np.nan, inplace=True)

        # Return the updated DataFrame
        return X

    def day_since_enrollment(self,data):
        # Convert 'Dt_Customer' to a date-time format
        data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y')

        # Find the max and min dates in the Dt_Customer column
        max_date = data['Dt_Customer'].max()
        min_date = data['Dt_Customer'].min()
        
        
        logging.info(f"The newest customer's enrolment date in the records:", max_date)
        logging.info("The oldest customer's enrolment date in the records:", min_date)

        # Calculate the number of days between the enrolment date and the maximum date
        data['Days_since_enrollment'] = (max_date - data['Dt_Customer']).dt.days

        return data

    
    
    def add_columns(self,df, column_names, new_column_name):
        df[new_column_name] = df[column_names].sum(axis=1)
        return df

    
    # Promotion dataframe 

    
        
    def run_data_modification(self,data):
        
        X=data.copy()
        
        # Drop na from dataframe
        X.dropna(inplace=True)
    
        # Removing duplicated rows 
        X=self.remove_duplicate_rows_keep_last(X)

        # make Null as np.nan
        X=self.convert_nan_null_to_nan(X)
        
        # Drop rows with nan
        X=self.drop_rows_with_nan(X)
        
        # Consumer Data 
        logging.info(" ------ Modifying consumer Data ------")
        # Total Offspring
        logging.info(" Creating Total Offspring column")
        X['Total_Offsprings'] = X['Kidhome'] + X['Teenhome']

        # Living With
        # Deriving Living attributes based on the marital status
        logging.info(" Creating Living_With column")
        X['Living_With'] = X['Marital_Status'].replace({'Married': 'Partner', 'Together': 'Partner', 'Single': 'Alone',
                                                        'Divorced': 'Alone', 'Widow': 'Alone', 'Absurd': 'Alone',
                                                        'YOLO': 'Alone'})

        # Feature indicating Family size
        logging.info(" Creating Family_Size column")
        X['Family_Size'] = X['Living_With'].replace({'Partner': 2, 'Alone': 1}) + X['Total_Offsprings']

        # Education column
        # Segment education levels in three groups
        logging.info(" Modifying  Education column")
        X['Education'] = X['Education'].replace({'Basic': 'Undergraduate',
                                                '2n Cycle': 'Undergraduate',
                                                'Graduation': 'Graduate',
                                                'Master': 'Postgraduate',
                                                'PhD': 'Postgraduate'})

        # Creating column Day since enrollment
        logging.info(" Creating Day_since_enrollment column")
        X = self.day_since_enrollment(X)

        # Creating Age column
        # Calculate the age of customers
        logging.info(" Creating Age column")
        X['Age'] = 2023 - X['Year_Birth']
        
        
        logging.info(" ------ Consumer Data Modified -------- ")
        
        # Product Data 
        column_names = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        X = self.add_columns(X,new_column_name="Total_products",column_names=column_names)
        
        # Rename columns for clarity and ease of use
        X = X.rename(columns={
            'MntWines': 'Wines',
            'MntFruits': 'Fruits',
            'MntMeatProducts': 'Meat',
            'MntFishProducts': 'Fish',
            'MntSweetProducts': 'Sweets',
            'MntGoldProds': 'Gold'
        })
        
        # Promotion Data 
        X['AcceptedCmp1'] = X['AcceptedCmp1'].replace(1, 1)
        X['AcceptedCmp2'] = X['AcceptedCmp2'].replace(1, 2)
        X['AcceptedCmp3'] = X['AcceptedCmp3'].replace(1, 3)
        X['AcceptedCmp4'] = X['AcceptedCmp4'].replace(1, 4)
        X['AcceptedCmp5'] = X['AcceptedCmp5'].replace(1, 5)
        X['Response'] = X['Response'].replace(1, 6)
        
        columns=['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3','AcceptedCmp4', 'AcceptedCmp5', 'Response']
        X=self.add_columns(X,columns,new_column_name='Frequency')
        
        
        
        # Place dataframe 
        X=self.add_columns(X,new_column_name='Total_purchases',column_names=['NumWebPurchases','NumCatalogPurchases','NumStorePurchases'])
        
        

        
        # Drop Columns 
        X=self.drop_columns(X=X)
        
        logging.info(f" Columne after dropping : {X.columns}")
        
        
        
        return X
    
    
    
    
    
    
    
                                            ######################### Outiers ############################
    

    
    def outlier(self,X):
        
        X = X[(X['Income'] < 600000)]

        return X
    

    
    
    def data_wrangling(self,X:pd.DataFrame):
        try:

            
            # Data Modification 
            data_modified=self.run_data_modification(data=X)
            
            logging.info(" Data Modification Done")
            
            # Removing outliers 
            
            logging.info(" Removing Outliers")
            
            df_outlier_removed=self.outlier(X=data_modified)
            
            
            
            
            
            return df_outlier_removed
    
        
        except Exception as e:
            raise ApplicationException(e,sys) from e
        
        
    
    
    
    
    def fit(self,X,y=None):
        return self
    
    
    def transform(self,X:pd.DataFrame,y=None):
        try:    
            data_modified=self.data_wrangling(X)

            
            
            
            logging.info(f"Original Data  : {X.shape}")
            logging.info(f"Shapde Modified Data : {data_modified.shape}")
            
            
            
            # Ecoding categorical features
            education_mapping = {
                'Graduate': 3,
                'Postgraduate': 2,
                'Undergraduate': 1
            }

            data_modified['Education'] = data_modified['Education'].map(education_mapping).astype('int')
            
            data_modified['Living_With'] = data_modified['Living_With'].map({'Alone': 0, 'Partner': 1}).astype(int)
            
    
         
            data_modified.to_csv("data_modified.csv",index=False)
            logging.info(" Data Wrangaling Done ")

            return data_modified
        except Exception as e:
            raise ApplicationException(e,sys) from e







class DataTransformation:
    
    
    def __init__(self, data_transformation_config: DataTransformationConfig,
                    data_validation_artifact: DataValidationArtifact):
        try:
            logging.info(f"\n{'*'*20} Data Transformation log started {'*'*20}\n\n")
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            
                                ############### Accesssing Column Labels #########################
                                
                                
                                #           Schema.yaml -----> DataTransfomation 
            
            # Transformation Yaml File path 
            
            # Reading data in Schema 
            self.transformation_yaml = read_yaml_file(file_path=TRANSFORMATION_YAML_FILE_PATH)

            self.drop_columns=self.transformation_yaml[DROP_COLUMNS]
          
           # self.drop_columns=self.schema[DROP_COLUMN_KEY]
            
                                ########################################################################
        except Exception as e:
            raise ApplicationException(e,sys) from e



    def get_feature_engineering_object(self):
        try:
            
            feature_engineering = Pipeline(steps = [("fe",Feature_Engineering(drop_columns=self.drop_columns))])
            return feature_engineering
        except Exception as e:
            raise ApplicationException(e,sys) from e
   
   
    def get_data_transformer_object(self,df):
        try:
            logging.info('Creating Data Transformer Object')
            numerical_col=df.columns
            numerical_pipeline = Pipeline(steps=[
                ('scaler', StandardScaler()),
            ])
            preprocessor = ColumnTransformer([
                ('numerical_pipeline', numerical_pipeline, numerical_col)
            ])
            return preprocessor


        except Exception as e:
                logging.error('An error occurred during data transformation')
                raise ApplicationException(e, sys) from e
            




    def initiate_data_transformation(self):
        try:
            # Data validation Artifact ------>Accessing train and test files 
            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_validation_artifact.validated_train_path
      

            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df = pd.read_csv(train_file_path,sep='\t')
         
            
            
            logging.info(f" Accesig train and test data \
                         Train Data : {train_file_path}" )
                       
            
            logging.info(f" Traning columns {train_df.columns}")

                        
            
            # Feature Engineering 
            logging.info(f"Obtaining feature engineering object.")
            fe_obj = self.get_feature_engineering_object()
            
            logging.info(f"Applying feature engineering object on training dataframe and testing dataframe")
            logging.info(">>>" * 20 + " Training data " + "<<<" * 20)
            logging.info(f"Feature Enineering - Train Data ")
            train_df = fe_obj.fit_transform(train_df)

            

            #logging.info(f" Columns in feature enginering {feature_eng_test_df.columns}")
            logging.info(f"Saving feature engineered training  dataframe.")
            transformed_train_dir = self.data_transformation_config.transformed_train_dir

            Feature_eng_train_file_path = os.path.join(transformed_train_dir,"Feature_engineering.csv")
            
            save_data(file_path = Feature_eng_train_file_path, data = train_df)
            
            
                        
                                    ############ Input Fatures transformation########
            ## Preprocessing 
            logging.info("*" * 20 + " Applying preprocessing object on training dataframe  " + "*" * 20)
            preprocessing_obj = self.get_data_transformer_object(df=train_df)
            train_arr = preprocessing_obj.fit_transform(train_df)
            # Log the shape of train_arr
            logging.info(f"Shape of train_arr: {train_arr.shape}")

            logging.info("Transformation completed successfully")
            
            col =train_df.columns
            
            transformed_train_df = pd.DataFrame(train_arr, columns=col )
    
 
            
            # Saving transformed data 
            transformed_train_dir = self.data_transformation_config.transformed_train_dir

            transformed_train_file_path = os.path.join(transformed_train_dir,"transformed_train.csv")


                    

                                ###############################################################
            
            # Saving the Transformed array ----> csv 
            ## Saving transformed train  file
            logging.info("Saving Transformed Train file")
            
            save_data(file_path = transformed_train_file_path, data = transformed_train_df)
            
            logging.info("Transformed Train file saved")
            logging.info("Saving Feature Engineering Object")
            
            ### Saving FFeature engineering and preprocessor object 
            logging.info("Saving Feature Engineering Object")
            feature_engineering_object_file_path = self.data_transformation_config.feature_engineering_object_file_path
            save_object(file_path = feature_engineering_object_file_path,obj = fe_obj)
            save_object(file_path=os.path.join(ROOT_DIR,PIKLE_FOLDER_NAME_KEY,
                                 os.path.basename(feature_engineering_object_file_path)),obj=fe_obj)

            logging.info("Saving Preprocessing Object")
            preprocessing_object_file_path = self.data_transformation_config.preprocessed_object_file_path
            save_object(file_path = preprocessing_object_file_path, obj = preprocessing_obj)
            save_object(file_path=os.path.join(ROOT_DIR,PIKLE_FOLDER_NAME_KEY,
                                 os.path.basename(preprocessing_object_file_path)),obj=preprocessing_obj)

            # Feature_eng_train_file_path
            Feature_eng_train_file_path=Feature_eng_train_file_path


            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
            message="Data transformation successfull.",
            Feature_eng_train_file_path=Feature_eng_train_file_path,
            transformed_train_file_path = transformed_train_file_path,
            preprocessed_object_file_path = preprocessing_object_file_path,
            feature_engineering_object_file_path = feature_engineering_object_file_path)
            
            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def __del__(self):
        logging.info(f"\n{'*'*20} Data Transformation log completed {'*'*20}\n\n")