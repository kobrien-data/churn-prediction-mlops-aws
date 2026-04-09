import os
import boto3
from datetime import datetime

client = boto3.client('sagemaker', region_name='eu-north-1')

role_arn = os.environ.get('SAGEMAKER_ROLE_ARN')
training_image_uri = os.environ.get('TRAINING_IMAGE_URI')
mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')

response = client.create_training_job(
    TrainingJobName=f"customer-churn-{datetime.now().strftime('%Y-%m-%d-%H%M')}",
    RoleArn=role_arn,
    AlgorithmSpecification={
        'TrainingImage': training_image_uri,
        'TrainingInputMode': 'File',
    },
    ResourceConfig={
        'InstanceType': 'ml.m5.xlarge',
        'InstanceCount': 1,
        'VolumeSizeInGB': 30,
    },
    InputDataConfig=[
        {
            'ChannelName': 'train',
            'DataSource': {
                'S3DataSource': {
                    'S3Uri': f"s3://{os.environ.get('S3_PROCESSED_DATA_BUCKET')}/train/",
                    'S3DataType': 'S3Prefix',
                }
            },
            'ContentType': 'text/csv',
        },
        {
            'ChannelName': 'test',
            'DataSource': {
                'S3DataSource': {
                    'S3Uri': f"s3://{os.environ.get('S3_PROCESSED_DATA_BUCKET')}/test/",
                    'S3DataType': 'S3Prefix',
                }
            },
            'ContentType': 'text/csv',
        },
    ],
    OutputDataConfig={
        'S3OutputPath': f"s3://{os.environ.get('S3_MODEL_ARTIFACTS_BUCKET')}/",
    },
    StoppingCondition={
        'MaxRuntimeInSeconds': 3600,
    },
    Environment={
        'MLFLOW_TRACKING_URI': mlflow_tracking_uri,
    },
)

print(response)
