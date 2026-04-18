# -----------------------------------------------------
# SNS Topic — notifies when drift is detected
# -----------------------------------------------------

resource "aws_sns_topic" "retraining_trigger" {
  name = "churn-retraining-topic"

  tags = {
    Project   = "customer-churn-mlops"
    ManagedBy = "terraform"
  }
}

# -----------------------------------------------------
# CloudWatch Alarm — fires when drift exceeds threshold
# -----------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "model_drift" {
  alarm_name          = "churn-model-drift-alarm"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "feature_baseline_drift"
  namespace           = "/aws/sagemaker/Endpoints/data-metrics"
  period              = 900
  statistic           = "Average"
  threshold           = 0.5
  alarm_description   = "Triggers retraining when feature drift is detected"

  alarm_actions = [aws_sns_topic.retraining_trigger.arn]

  tags = {
    Project   = "customer-churn-mlops"
    ManagedBy = "terraform"
  }
}

# -----------------------------------------------------
# IAM Role — allows EventBridge to trigger SageMaker
# -----------------------------------------------------

resource "aws_iam_role" "eventbridge_sagemaker" {
  name = "churn-eventbridge-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "events.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = {
    Project   = "customer-churn-mlops"
    ManagedBy = "terraform"
  }
}

resource "aws_iam_role_policy" "eventbridge_sagemaker" {
  name = "churn-eventbridge-sagemaker-policy"
  role = aws_iam_role.eventbridge_sagemaker.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "sagemaker:StartPipelineExecution"
      Resource = "arn:aws:sagemaker:*:*:pipeline/customer-churn-pipeline"
    }]
  })
}

# -----------------------------------------------------
# EventBridge Rule — listens for CloudWatch alarm state
# -----------------------------------------------------

resource "aws_cloudwatch_event_rule" "retraining_trigger" {
  name        = "churn-retraining-trigger"
  description = "Triggers retraining pipeline when drift alarm fires"

  event_pattern = jsonencode({
    source      = ["aws.cloudwatch"]
    detail-type = ["CloudWatch Alarm State Change"]
    detail = {
      alarmName = ["churn-model-drift-alarm"]
      state = {
        value = ["ALARM"]
      }
    }
  })

  tags = {
    Project   = "customer-churn-mlops"
    ManagedBy = "terraform"
  }
}

# -----------------------------------------------------
# EventBridge Target — wires the rule to the pipeline
# -----------------------------------------------------

resource "aws_cloudwatch_event_target" "sagemaker_pipeline" {
  rule     = aws_cloudwatch_event_rule.retraining_trigger.name
  arn      = "arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:pipeline/customer-churn-pipeline"
  role_arn = aws_iam_role.eventbridge_sagemaker.arn

  sagemaker_pipeline_target {
    pipeline_parameter_list {
      name  = "dummy"
      value = "placeholder"
    }
  }
}

# -----------------------------------------------------
# Outputs
# -----------------------------------------------------

output "drift_alarm_name" {
  description = "CloudWatch alarm name for drift detection"
  value       = aws_cloudwatch_metric_alarm.model_drift.alarm_name
}

output "retraining_topic_arn" {
  description = "SNS topic ARN for retraining notifications"
  value       = aws_sns_topic.retraining_trigger.arn
}