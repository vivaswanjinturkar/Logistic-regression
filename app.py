#from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import uvicorn
from typing import Dict,List
sys.path.append(str(Path(__file__).parent.parent))
from src.components.pipeline.predict_pipeline import PredictPipeline

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Json
import pandas as pd

app = FastAPI()

class BankData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float

class MultipleBankData(BaseModel):
    data: List[BankData]  # List of BankData instances


@app.post("/predict")
async def process_multiple_data(data: MultipleBankData):
    data=[i.dict() for i in data.data]
    df = pd.DataFrame(data)
    print(df)
    df.rename(
        {
            'nr_employed': 'nr.employed',
            'cons_price_idx': 'cons.price.idx',
            'cons_conf_idx': 'cons.conf.idx',
            'emp_var_rate': 'emp.var.rate',
        },
        axis=1,
        inplace=True,
    )
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(df)
    # print(results)
    # print(type(results))
    df['prediction']=results
    return {"predictions": df.to_dict(orient="records")}



# if __name__=="__main__":  
#     uvicorn.run(app, host="0.0.0.0", port=8889,workers=4)

      

