import sys
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.exception import CustomException
from src.utils import load_object
from src.custom_logger import logging
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            ordinal_encoded_features = ['marital','default','housing','loan','poutcome','contact']
            one_hot_encoded_features = ['job']
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


            print("features",features)
            features['month']=features['month'].map({'dec':12,'may':5, 'feb':2, 'nov':11, 'oct':10, 'sep':9, 'mar':3, 'apr':4,
            'aug':8, 'jun':6, 'jan':1, 'jul':7})
            features['day_of_week']=features['day_of_week'].map({'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5})
            features['education']=features['education'].map({'basic.4y':2, 'high.school':5, 'basic.6y':3, 'basic.9y':4,
                    'professional.course':6, 'unknown':1, 'university.degree':7,
                    'illiterate':0})
            features.drop('duration',inplace=True,axis=1)
            transformed_features=preprocessor.transform(features)
            
            one_hot_column_names = preprocessor.named_transformers_['one_hot_encoder'].get_feature_names_out(one_hot_encoded_features)

            # Fit the pipeline on your data
            transformed_features = pd.DataFrame(transformed_features, columns=ordinal_encoded_features + list(one_hot_column_names) + numerical_columns)
            preds=model.predict(transformed_features)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

