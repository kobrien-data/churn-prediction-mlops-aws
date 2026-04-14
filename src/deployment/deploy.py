import os
import boto3
from datetime import datetime, timezone

REGION = 'eu-north-1'
MODEL_PACKAGE_GROUP = 'customer-churn-models'
ENDPOINT_NAME = 'customer-churn-endpoint'
INSTANCE_TYPE = 'ml.m5.xlarge'
INFERENCE_IMAGE_URI = '941377133770.dkr.ecr.eu-north-1.amazonaws.com/customer-churn-inference:latest'

role = os.environ['SAGEMAKER_ROLE_ARN']

sm = boto3.client('sagemaker', region_name=REGION)


def get_latest_approved_model_package() -> str:
    """Return the ARN of the most recently created Approved model package."""
    paginator = sm.get_paginator('list_model_packages')
    pages = paginator.paginate(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        ModelApprovalStatus='Approved',
        SortBy='CreationTime',
        SortOrder='Descending',
    )
    for page in pages:
        packages = page['ModelPackageSummaryList']
        if packages:
            arn = packages[0]['ModelPackageArn']
            print(f'Found approved model package: {arn}')
            return arn
    raise RuntimeError(f'No Approved model packages found in group: {MODEL_PACKAGE_GROUP}')


def create_model(model_name: str, model_package_arn: str) -> None:
    model_package = sm.describe_model_package(ModelPackageName=model_package_arn)
    model_data_url = model_package['InferenceSpecification']['Containers'][0]['ModelDataUrl']
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'Image': INFERENCE_IMAGE_URI,
            'ModelDataUrl': model_data_url,
            'Environment': {'SAGEMAKER_PROGRAM': 'inference.py'},
        },
        ExecutionRoleArn=role,
    )
    print(f'Created model: {model_name}')


def create_endpoint_config(config_name: str, model_name: str) -> None:
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                'VariantName': 'primary',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': INSTANCE_TYPE,
                'InitialVariantWeight': 1.0,
            }
        ],
    )
    print(f'Created endpoint config: {config_name}')


def deploy_endpoint(endpoint_config_name: str) -> None:
    """Create the endpoint if it doesn't exist, otherwise update it."""
    try:
        sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        print(f'Endpoint {ENDPOINT_NAME!r} exists — updating...')
        sm.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=endpoint_config_name,
        )
    except sm.exceptions.ClientError:
        print(f'Creating endpoint {ENDPOINT_NAME!r}...')
        sm.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=endpoint_config_name,
        )

    waiter = sm.get_waiter('endpoint_in_service')
    print('Waiting for endpoint to be InService...')
    waiter.wait(EndpointName=ENDPOINT_NAME, WaiterConfig={'Delay': 30, 'MaxAttempts': 40})
    print(f'Endpoint {ENDPOINT_NAME!r} is InService.')


def main():
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
    model_name = f'customer-churn-{timestamp}'
    endpoint_config_name = f'customer-churn-config-{timestamp}'

    model_package_arn = get_latest_approved_model_package()
    create_model(model_name, model_package_arn)
    create_endpoint_config(endpoint_config_name, model_name)
    deploy_endpoint(endpoint_config_name)


if __name__ == '__main__':
    main()
