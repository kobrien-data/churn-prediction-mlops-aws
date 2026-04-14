import json
import os

import joblib
import pandas as pd


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, 'model.joblib'))


def input_fn(request_body, content_type):
    if content_type != 'application/json':
        raise ValueError(f'Unsupported content type: {content_type}')
    data = json.loads(request_body)
    return pd.DataFrame([data])


def predict_fn(input_data, model):
    prediction = int(model.predict(input_data)[0])
    churn_probability = float(model.predict_proba(input_data)[0, 1])
    return {'prediction': prediction, 'churn_probability': churn_probability}


def output_fn(prediction, accept):
    return json.dumps(prediction), 'application/json'
