import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,    OrdinalEncoder

from src.exception import CustomException
from src.logger import logging
import os

from utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:

            ordinal_encoded_features = ['marital','default','housing','loan','poutcome','contact']
            one_hot_encoded_features = ['job']
            categorical_columns=ordinal_encoded_features+one_hot_encoded_features
            numerical_columns=['age',
                    'education',
                    'month',
                    'day_of_week',
                    'campaign',
                    'pdays',
                    'previous',
                    'emp.var.rate',
                    'cons.price.idx',
                    'cons.conf.idx',
                    'euribor3m',
                    'nr.employed']

            # Create transformers
            ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
            one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

            # Create a column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('ordinal_encoder', ordinal_encoder, ordinal_encoded_features),
                    ('one_hot_encoder', one_hot_encoder, one_hot_encoded_features)
                ],
                remainder='passthrough'  # Pass through the columns not specified in transformers
            )


            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessor=self.get_data_transformer_object()

            target_column_name="y"
            ordinal_encoded_features = ['marital','default','housing','loan','poutcome','contact']
            one_hot_encoded_features = ['job']
            categorical_columns=ordinal_encoded_features+one_hot_encoded_features
            numerical_columns=['age',
                    'education',
                    'month',
                    'day_of_week',
                    'campaign',
                    'pdays',
                    'previous',
                    'emp.var.rate',
                    'cons.price.idx',
                    'cons.conf.idx',
                    'euribor3m',
                    'nr.employed']

            train_df['month']=train_df['month'].map({'dec':12,'may':5, 'feb':2, 'nov':11, 'oct':10, 'sep':9, 'mar':3, 'apr':4,
            'aug':8, 'jun':6, 'jan':1, 'jul':7})
            train_df['day_of_week']=train_df['day_of_week'].map({'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5})
            train_df['education']=train_df['education'].map({'basic.4y':2, 'high.school':5, 'basic.6y':3, 'basic.9y':4,
                    'professional.course':6, 'unknown':1, 'university.degree':7,
                    'illiterate':0})
            train_df.drop('duration',inplace=True,axis=1)

            test_df['month']=test_df['month'].map({'dec':12,'may':5, 'feb':2, 'nov':11, 'oct':10, 'sep':9, 'mar':3, 'apr':4,
            'aug':8, 'jun':6, 'jan':1, 'jul':7})
            test_df['day_of_week']=test_df['day_of_week'].map({'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5})
            test_df['education']=test_df['education'].map({'basic.4y':2, 'high.school':5, 'basic.6y':3, 'basic.9y':4,
                    'professional.course':6, 'unknown':1, 'university.degree':7,
                    'illiterate':0})
            test_df.drop('duration',inplace=True,axis=1)


            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )


            transformed_data_train = preprocessor.fit_transform(input_feature_train_df)
            transformed_data_test = preprocessor.transform(input_feature_test_df)

            # Get the column names after one-hot encoding
            one_hot_column_names = preprocessor.named_transformers_['one_hot_encoder'].get_feature_names_out(one_hot_encoded_features)

            # Fit the pipeline on your data
            transformed_data_train = pd.DataFrame(transformed_data_train, columns=ordinal_encoded_features + list(one_hot_column_names) + numerical_columns)
            transformed_data_test = pd.DataFrame(transformed_data_test, columns=ordinal_encoded_features + list(one_hot_column_names) + numerical_columns)



            train_arr = [
                 transformed_data_train, np.array(target_feature_train_df)
             ]
            test_arr = [transformed_data_test, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)