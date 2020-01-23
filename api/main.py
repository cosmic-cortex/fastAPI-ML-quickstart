import numpy as np

from typing import List

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, ValidationError, validator

from .ml.model import load_model, n_features


class PredictRequest(BaseModel):
    data: List[List[float]]

    @validator('data')
    def check_dimensionality(cls, v):
        for point in v:
            if len(point) != n_features:
                raise ValueError(f"Each data point must contain {n_features} features")

        return v


class PredictResponse(BaseModel):
    data: List[float]


app = FastAPI()
model = load_model()


@app.post('/predict', response_model=PredictResponse)
def predict(input: PredictRequest):
    X = np.array(input.data)
    y_pred = model.predict(X)
    result = PredictResponse(data=y_pred.tolist())
    return result


@app.post('/predict-csv')
def predict_csv(csv_file: UploadFile = File(...)):
    import pdb
    pdb.set_trace()
