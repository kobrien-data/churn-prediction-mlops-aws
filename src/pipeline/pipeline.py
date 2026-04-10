import os
import boto3
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.inputs import TrainingInput

role = os.environ.get('SAGEMAKER_ROLE_ARN')
region = 'eu-north-1'
training_image_uri = os.environ.get('TRAINING_IMAGE_URI')
raw_data_bucket = os.environ.get('S3_RAW_DATA_BUCKET')
processed_data_bucket = os.environ.get('S3_PROCESSED_DATA_BUCKET')
model_artifacts_bucket = os.environ.get('S3_MODEL_ARTIFACTS_BUCKET')
mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')

session = PipelineSession(boto_session=boto3.Session(region_name=region))

cache_config = CacheConfig(enable_caching=True, expire_after="7d")

# Processing Step — runs preprocessing.py on raw data
processing_processor = ScriptProcessor(
    image_uri=training_image_uri,
    command=['python3'],
    instance_type='ml.t3.medium',
    instance_count=1,
    role=role,
    sagemaker_session=session
)

processing_step = ProcessingStep(
    name='preprocess-data',
    processor=processing_processor,
    inputs=[
        ProcessingInput(
            source=f's3://{raw_data_bucket}/Customer-Churn-Records.csv',
            destination='/opt/ml/processing/input'
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name='train',
            source='/opt/ml/processing/output/train',
            destination=f's3://{processed_data_bucket}/train'
        ),
        ProcessingOutput(
            output_name='test',
            source='/opt/ml/processing/output/test',
            destination=f's3://{processed_data_bucket}/test'
        )
    ],
    code='src/data/preprocessing.py',
    cache_config=cache_config
)

# Training Step — runs train.py on preprocessed data
estimator = Estimator(
    image_uri=training_image_uri,
    role=role,
    instance_type='ml.m5.xlarge',
    instance_count=1,
    volume_size=30,
    output_path=f's3://{model_artifacts_bucket}/',
    environment={'MLFLOW_TRACKING_URI': mlflow_tracking_uri},
    container_entry_point=['python', 'src/training/train.py'],
    sagemaker_session=session
)

training_step = TrainingStep(
    name='train-model',
    estimator=estimator,
    inputs={
        'train': TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri,
            content_type='text/csv'
        ),
        'test': TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs['test'].S3Output.S3Uri,
            content_type='text/csv'
        )
    },
    cache_config=cache_config
)

# Evaluation Step — runs evaluate.py and writes metrics.json
evaluation_processor = ScriptProcessor(
    image_uri=training_image_uri,
    command=['python3'],
    instance_type='ml.t3.medium',
    instance_count=1,
    role=role,
    env={'MLFLOW_TRACKING_URI': mlflow_tracking_uri},
    sagemaker_session=session
)

evaluation_report = PropertyFile(
    name='evaluation-report',
    output_name='metrics',
    path='metrics.json'
)

evaluation_step = ProcessingStep(
    name='evaluate-model',
    processor=evaluation_processor,
    inputs=[
        ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination='/opt/ml/processing/model'
        ),
        ProcessingInput(
            source=processing_step.properties.ProcessingOutputConfig.Outputs['test'].S3Output.S3Uri,
            destination='/opt/ml/processing/test'
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name='metrics',
            source='/opt/ml/processing/output/metrics',
            destination=f's3://{model_artifacts_bucket}/evaluation'
        )
    ],
    code='src/evaluation/evaluate.py',
    property_files=[evaluation_report],
    cache_config=cache_config
)

# Condition Step — only register if AUC >= 0.75
auc_condition = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=evaluation_step.name,
        property_file=evaluation_report,
        json_path='roc_auc_score'
    ),
    right=0.75
)

# Register Step — registers model in SageMaker Model Registry
model = Model(
    image_uri=training_image_uri,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    sagemaker_session=session
)

model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=f's3://{model_artifacts_bucket}/evaluation/metrics.json',
        content_type='application/json'
    )
)

register_step = ModelStep(
    name='register-model',
    step_args=model.register(
        content_types=['text/csv'],
        response_types=['application/json'],
        inference_instances=['ml.m5.xlarge'],
        transform_instances=['ml.m5.xlarge'],
        model_package_group_name='customer-churn-models',
        approval_status='Approved',
        model_metrics=model_metrics
    )
)

condition_step = ConditionStep(
    name='check-auc-threshold',
    conditions=[auc_condition],
    if_steps=[register_step],
    else_steps=[]
)

# Pipeline definition
pipeline = Pipeline(
    name='customer-churn-pipeline',
    steps=[processing_step, training_step, evaluation_step, condition_step],
    sagemaker_session=session
)

if __name__ == '__main__':
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    print(f'Pipeline started: {execution.arn}')
