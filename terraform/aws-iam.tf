locals {
  name_prefix = "customer-churn"
}

# -----------------------------------------------------
# Trust Policy — allows SageMaker to assume this role
# -----------------------------------------------------

data "aws_iam_policy_document" "sagemaker_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

# -----------------------------------------------------
# SageMaker Execution Role
# -----------------------------------------------------

resource "aws_iam_role" "sagemaker_execution_role" {
  name               = "${local.name_prefix}-sagemaker-execution-role"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume_role.json

  tags = {
    Project     = "customer-churn-mlops"
    ManagedBy   = "terraform"
  }
}

# -----------------------------------------------------
# S3 Policy — scoped to project buckets only
# -----------------------------------------------------

data "aws_iam_policy_document" "sagemaker_s3" {
  statement {
    sid    = "AllowProjectBucketAccess"
    effect = "Allow"

    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
    ]

    resources = [
      "arn:aws:s3:::customer-churn-raw-data",
      "arn:aws:s3:::customer-churn-raw-data/*",
      "arn:aws:s3:::customer-churn-processed-data",
      "arn:aws:s3:::customer-churn-processed-data/*",
      "arn:aws:s3:::customer-churn-model-artifacts",
      "arn:aws:s3:::customer-churn-model-artifacts/*",
    ]
  }
}

resource "aws_iam_policy" "sagemaker_s3" {
  name        = "${local.name_prefix}-sagemaker-s3-policy"
  description = "Scoped S3 access for SageMaker — project buckets only"
  policy      = data.aws_iam_policy_document.sagemaker_s3.json
}

resource "aws_iam_role_policy_attachment" "sagemaker_s3" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = aws_iam_policy.sagemaker_s3.arn
}

# -----------------------------------------------------
# ECR Policy — pull training images only
# -----------------------------------------------------

data "aws_iam_policy_document" "sagemaker_ecr" {
  statement {
    sid    = "AllowECRAuth"
    effect = "Allow"

    actions = [
      "ecr:GetAuthorizationToken",
    ]

    resources = ["*"] # GetAuthorizationToken cannot be scoped to a resource
  }

  statement {
    sid    = "AllowECRPull"
    effect = "Allow"

    actions = [
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:BatchCheckLayerAvailability",
    ]

    resources = [
      "arn:aws:ecr:*:*:repository/customer-churn-*",
    ]
  }
}

resource "aws_iam_policy" "sagemaker_ecr" {
  name        = "${local.name_prefix}-sagemaker-ecr-policy"
  description = "Scoped ECR access for SageMaker — pull from project repositories only"
  policy      = data.aws_iam_policy_document.sagemaker_ecr.json
}

resource "aws_iam_role_policy_attachment" "sagemaker_ecr" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = aws_iam_policy.sagemaker_ecr.arn
}

# -----------------------------------------------------
# CloudWatch Logs Policy — write training job logs only
# -----------------------------------------------------

data "aws_iam_policy_document" "sagemaker_cloudwatch" {
  statement {
    sid    = "AllowCloudWatchLogs"
    effect = "Allow"

    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
    ]

    resources = [
      "arn:aws:logs:*:*:log-group:/aws/sagemaker/*",
    ]
  }

  statement {
    sid    = "AllowCloudWatchMetrics"
    effect = "Allow"

    actions = [
      "cloudwatch:PutMetricData",
    ]

    resources = ["*"] # PutMetricData cannot be scoped to a resource
  }
}

resource "aws_iam_policy" "sagemaker_cloudwatch" {
  name        = "${local.name_prefix}-sagemaker-cloudwatch-policy"
  description = "CloudWatch logs and metrics access for SageMaker training jobs"
  policy      = data.aws_iam_policy_document.sagemaker_cloudwatch.json
}

resource "aws_iam_role_policy_attachment" "sagemaker_cloudwatch" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = aws_iam_policy.sagemaker_cloudwatch.arn
}

# -----------------------------------------------------
# SageMaker Pipeline Policy — run and manage pipelines
# -----------------------------------------------------

data "aws_iam_policy_document" "sagemaker_pipelines" {
  statement {
    sid    = "AllowSageMakerPipelines"
    effect = "Allow"

    actions = [
      "sagemaker:CreatePipeline",
      "sagemaker:UpdatePipeline",
      "sagemaker:DeletePipeline",
      "sagemaker:StartPipelineExecution",
      "sagemaker:StopPipelineExecution",
      "sagemaker:DescribePipeline",
      "sagemaker:DescribePipelineExecution",
      "sagemaker:ListPipelineExecutionSteps",
      "sagemaker:CreateProcessingJob",
      "sagemaker:CreateTrainingJob",
      "sagemaker:CreateModel",
      "sagemaker:CreateModelPackage",
      "sagemaker:DescribeTrainingJob",
      "sagemaker:DescribeProcessingJob",
      "sagemaker:DescribeModelPackage",
      "sagemaker:UpdateModelPackage",
      "sagemaker:CreateEndpoint",
      "sagemaker:CreateEndpointConfig",
      "sagemaker:UpdateEndpoint",
      "sagemaker:DescribeEndpoint",
      "sagemaker:InvokeEndpoint",
    ]

    resources = [
      "arn:aws:sagemaker:*:*:pipeline/customer-churn-*",
      "arn:aws:sagemaker:*:*:training-job/customer-churn-*",
      "arn:aws:sagemaker:*:*:processing-job/customer-churn-*",
      "arn:aws:sagemaker:*:*:model/customer-churn-*",
      "arn:aws:sagemaker:*:*:model-package/customer-churn-*",
      "arn:aws:sagemaker:*:*:endpoint/customer-churn-*",
      "arn:aws:sagemaker:*:*:endpoint-config/customer-churn-*",
    ]
  }
}

resource "aws_iam_policy" "sagemaker_pipelines" {
  name        = "${local.name_prefix}-sagemaker-pipelines-policy"
  description = "SageMaker pipeline and job permissions — scoped to project resources"
  policy      = data.aws_iam_policy_document.sagemaker_pipelines.json
}

resource "aws_iam_role_policy_attachment" "sagemaker_pipelines" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = aws_iam_policy.sagemaker_pipelines.arn
}

# -----------------------------------------------------
# Output — role ARN for use in other terraform files
# -----------------------------------------------------

output "sagemaker_execution_role_arn" {
  description = "ARN of the SageMaker execution role — reference this in sagemaker.tf"
  value       = aws_iam_role.sagemaker_execution_role.arn
}
