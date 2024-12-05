#Putri Fatiha Nuzula / 2602193042 / LC09

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()
model = joblib.load('bestmodel.pkl')

class input(BaseModel):
    age : int
    job : int
    # marital : int
    education : int
    default : int
    housing : int
    # loan : int
    contact : int 
    # month : int
    # day_of_week : int
    duration : float
    campaign : int
    days : int
    previous : int
    # outcome : int

@app.get("/")
def read_root():
    return {"messege": "Welcome to the Bank Revenue Model API"}

@app.post('/predict')
def predict(y: input):
    data = y.dict()
    features = [[data['age'], data['job'], data['education'], data['default'], data['housing'],
                data['contact'], data['duration'], data['campaign'],
                data['days'], data['previous']]]
    features_array = np.array(features)
    prediction = model.predict(features_array)
    return{'Prediction': prediction[0]}
