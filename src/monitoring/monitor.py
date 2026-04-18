from sagemaker.model_monitor import DefaultModelMonitor, CronExpressionGenerator
from sagemaker.model_monitor.dataset_format import DatasetFormat

DEFAULT_ROLE = "arn:aws:iam::941377133770:role/customer-churn-sagemaker-execution-role"
ENDPOINT_NAME = "customer-churn-endpoint"
BASELINE_DATA_URI = "s3://customer-churn-processed-data-941377133770/train/X_train.csv"
BASELINE_RESULTS_URI = "s3://customer-churn-monitoring-941377133770/baseline/"
DATA_CAPTURE_URI = "s3://customer-churn-model-artifacts-941377133770/data-capture/"
MONITORING_REPORTS_URI = "s3://customer-churn-monitoring-941377133770/reports/"

customer_churn_monitor = DefaultModelMonitor(
    role=DEFAULT_ROLE,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
)

customer_churn_monitor.suggest_baseline(
    baseline_dataset=BASELINE_DATA_URI,
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri=BASELINE_RESULTS_URI,
    wait=True,
)

customer_churn_monitor.create_monitoring_schedule(
    monitor_schedule_name="customer-churn-monitor-schedule",
    endpoint_input=ENDPOINT_NAME,
    output_s3_uri=MONITORING_REPORTS_URI,
    statistics=customer_churn_monitor.baseline_statistics(),
    constraints=customer_churn_monitor.suggested_constraints(),
    schedule_cron_expression=CronExpressionGenerator.hourly()
)
