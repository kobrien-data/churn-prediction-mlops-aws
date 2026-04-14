import json
import os

import boto3

sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=os.environ['AWS_REGION'])
ENDPOINT_NAME = os.environ['SAGEMAKER_ENDPOINT_NAME']


def handler(event, context):
    try:
        body = json.loads(event.get('body') or '{}')
    except json.JSONDecodeError:
        return {'statusCode': 400, 'body': json.dumps({'error': 'Invalid JSON body'})}

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        Body=json.dumps(body)
    )

    result = json.loads(response['Body'].read())
    prediction = result['prediction']
    churn_probability = result['churn_probability']

    # Confidence = probability of the predicted class
    confidence = churn_probability if prediction == 1 else 1 - churn_probability

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({
            'prediction': prediction,
            'churn': bool(prediction),
            'churn_probability': round(churn_probability, 4),
            'confidence': round(confidence, 4)
        })
    }
