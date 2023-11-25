import sys
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.exception import CustomException
from src.utils import load_object
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



class CustomData:
    def __init__(  self,
                age: float, job: str, marital: str, education: float, default: str, housing: str, loan: str, contact: str,
                 month: float, day_of_week: float, duration: float, campaign: float, pdays: float, previous: float, poutcome: str,
                 emp_var_rate: float, cons_price_idx: float, cons_conf_idx: float, euribor3m: float, nr_employed: float):

        self.age = age
        self.job = job
        self.marital = marital
        self.education = education
        self.default = default
        self.housing = housing
        self.loan = loan
        self.contact = contact
        self.month = month
        self.day_of_week = day_of_week
        self.duration = duration
        self.campaign = campaign
        self.pdays = pdays
        self.previous = previous
        self.poutcome = poutcome
        self.emp_var_rate = emp_var_rate
        self.cons_price_idx = cons_price_idx
        self.cons_conf_idx = cons_conf_idx
        self.euribor3m = euribor3m
        self.nr_employed = nr_employed

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
            "age": [self.age],
            "job": [self.job],
            "marital": [self.marital],
            "education": [self.education],
            "default": [self.default],
            "housing": [self.housing],
            "loan": [self.loan],
            "contact": [self.contact],
            "month": [self.month],
            "day_of_week": [self.day_of_week],
            "duration": [self.duration],
            "campaign": [self.campaign],
            "pdays": [self.pdays],
            "previous": [self.previous],
            "poutcome": [self.poutcome],
            "emp_var_rate": [self.emp_var_rate],
            "cons_price_idx": [self.cons_price_idx],
            "cons_conf_idx": [self.cons_conf_idx],
            "euribor3m": [self.euribor3m],
            "nr_employed": [self.nr_employed]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
