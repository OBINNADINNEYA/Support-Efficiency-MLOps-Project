import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from src.exceptions import CustomException
from src.logger import logging
from src.utils import *


@dataclass
class DataTransformationConfig:
    #preprocessing pickle file path assignment
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    encoder_obj_file_path = os.path.join('artifacts','labelencoder.pkl')


class DataTrasformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):

        """
        this function is responsible for creating the preprocessor for the numeric variables
        
        """
        try:

            num_cols  = ["text_size","comment_count","participants_count","first_response_minutes"]
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", RobustScaler(), num_cols),
                ],
            )
            
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
        
    def initiate_data_transformation(self,train_path, test_path):

        """
        this function initiates data transformation cleaned and transforms train and test data
          and saves the preprocessing object in a pockle file
        """

        try:

            train_df = build_model_table_from_parquet(train_path)
            test_df = build_model_table_from_parquet(test_path)

            logging.info("read and cleaned train and test data completed")

            logging.info("obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = ["resolution_class"]
            X_cols = ["text_size", "comment_count","participants_count","first_response_minutes","first_response_missing"]
            
            #pull out the target variable from the input variables for train and test
            input_feature_train_df = train_df[X_cols]
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df[X_cols]
            target_feature_test_df = test_df[target_column_name]

            #process target variable
            le = LabelEncoder()
            encodeded_target_feature_train_df = le.fit_transform(target_feature_train_df)   # y_train is df column of classes
            encodeded_target_feature_test_df  = le.transform(target_feature_test_df)


            logging.info("applying preprocessing object on train and test dfs")
            #transform the data 
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            #merge the transformed array with the target variables 
            train_arr = np.c_[input_feature_train_arr, np.array(encodeded_target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(encodeded_target_feature_test_df)]

            logging.info("saving preprocessing object ")
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info("saving labelencoder object ")
            save_object(
                file_path = self.data_transformation_config.encoder_obj_file_path,
                obj = le
            )

            logging.info("data transformation complete ")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.encoder_obj_file_path,

            )
        
            
        except Exception as e:
            raise CustomException(e,sys)



    

