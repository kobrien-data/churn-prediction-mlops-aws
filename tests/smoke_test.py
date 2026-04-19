import os
import requests

API_URL = os.environ.get(
    'API_GATEWAY_URL',
    'https://zovdxgkcg9.execute-api.eu-north-1.amazonaws.com/prod/predict'
)

KNOWN_INPUT = {
    "CreditScore": -0.4400359548576657,
    "Age": -0.08789693990845966,
    "Tenure": 1.03290776479748,
    "Balance": 0.4354184137433293,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 0.6432125326414329,
    "Satisfaction Score": -0.009816137054176019,
    "Point Earned": 1.0467938184725165,
    "Geography_France": 0,
    "Geography_Germany": 1,
    "Geography_Spain": 0,
    "Gender_Female": 1,
    "Gender_Male": 0,
    "Card Type_DIAMOND": 0,
    "Card Type_GOLD": 0,
    "Card Type_PLATINUM": 0,
    "Card Type_SILVER": 1
}


def test_endpoint_returns_200():
    response = requests.post(API_URL, json=KNOWN_INPUT)
    assert response.status_code == 200


def test_response_has_expected_keys():
    response = requests.post(API_URL, json=KNOWN_INPUT)
    body = response.json()
    assert 'prediction' in body
    assert 'churn' in body
    assert 'churn_probability' in body
    assert 'confidence' in body


def test_response_values_are_valid():
    response = requests.post(API_URL, json=KNOWN_INPUT)
    body = response.json()
    assert body['prediction'] in (0, 1)
    assert isinstance(body['churn'], bool)
    assert 0.0 <= body['churn_probability'] <= 1.0
    assert 0.0 <= body['confidence'] <= 1.0


def test_known_input_prediction():
    response = requests.post(API_URL, json=KNOWN_INPUT)
    body = response.json()
    assert body['prediction'] == 1
    assert body['churn'] is True
    assert body['churn_probability'] > 0.5
