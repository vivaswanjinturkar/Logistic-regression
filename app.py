from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import uvicorn
from typing import Dict,List
sys.path.append(str(Path(__file__).parent.parent))
from src.components.pipeline.predict_pipeline import CustomData,PredictPipeline

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Json
import pandas as pd

app = FastAPI()

class CustomDataInput(BaseModel):
    instances:List


@app.post("/predict")
async def process_custom_data(data: CustomDataInput):


    print(data.instances)
    # Create a Pandas DataFrame from the dictionary
    df = pd.DataFrame(data.instances)

    # Your further processing with the DataFrame can go here
    # For example, print the DataFrame to the console
    predict_pipeline=PredictPipeline()
    results=predict_pipeline.predict(df)
    df['prediction']=results
    return df.to_dict()


if __name__=="__main__":  
    uvicorn.run(app, host="0.0.0.0", port=8889)

      

