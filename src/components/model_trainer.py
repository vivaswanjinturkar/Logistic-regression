import os
import sys
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[0],
                train_array[1],
                test_array[0],
                test_array[1]
            )

            rfc=RandomForestClassifier(n_estimators=100,class_weight='balanced',max_depth=20)
            rfc.fit(X_train,y_train)
            rfc_prob=rfc.predict_proba(X_test)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=rfc
            )

            return roc_auc_score(y_test,rfc_prob[:,1])
            



            
        except Exception as e:
            raise CustomException(e,sys)