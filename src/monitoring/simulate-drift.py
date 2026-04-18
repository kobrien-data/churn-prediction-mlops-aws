import boto3
import json
import os
import numpy as np

runtime = boto3.client('sagemaker-runtime')

def send_shifted_data(endpoint_name, num_requests=100):
    for _ in range(num_requests):
        # Deliberately shift the feature values far outside
        # the training distribution to trigger drift detection
        shifted_payload = {
            'CreditScore': np.random.uniform(-4.0, -2.5),  # normally 300-850
            'Age': np.random.uniform(3.0, 5.0),            # normally 18-60
            'Tenure': np.random.uniform(-2.5, -1.5),
            'Balance': np.random.uniform(4.0, 6.0),        # normally lower
            'NumOfProducts': np.random.uniform(3.0, 4.0),  # normally 1-2
            'HasCrCard': 0,
            'IsActiveMember': 0,
            'EstimatedSalary': np.random.uniform(4.0, 6.0),
            'Satisfaction Score': np.random.uniform(-3.0, -2.0),
            'Point Earned': np.random.uniform(3.0, 5.0),
            'Geography_France': 0,
            'Geography_Germany': 1,
            'Geography_Spain': 0,
            'Gender_Female': 0,
            'Gender_Male': 1,
            'Card Type_DIAMOND': 0,
            'Card Type_GOLD': 0,
            'Card Type_PLATINUM': 0,
            'Card Type_SILVER': 1,
        }

        runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(shifted_payload)
        )

    print(f"Sent {num_requests} shifted requests to {endpoint_name}")

send_shifted_data(os.environ['SAGEMAKER_ENDPOINT_NAME'])
