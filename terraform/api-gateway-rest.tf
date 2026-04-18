# -----------------------------------------------------
# IAM role — allows API Gateway to write to CloudWatch
# -----------------------------------------------------

data "aws_iam_policy_document" "apigw_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["apigateway.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "apigw_cloudwatch" {
  name               = "customer-churn-apigw-cloudwatch-role"
  assume_role_policy = data.aws_iam_policy_document.apigw_assume_role.json

  tags = {
    Project   = "customer-churn-mlops"
    ManagedBy = "terraform"
  }
}

resource "aws_iam_role_policy_attachment" "apigw_cloudwatch" {
  role       = aws_iam_role.apigw_cloudwatch.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonAPIGatewayPushToCloudWatchLogs"
}

# Associates the CloudWatch role with the API Gateway account (account-wide setting)
resource "aws_api_gateway_account" "main" {
  cloudwatch_role_arn = aws_iam_role.apigw_cloudwatch.arn
}

# -----------------------------------------------------
# CloudWatch log group for access logs
# -----------------------------------------------------

resource "aws_cloudwatch_log_group" "apigw_access_logs" {
  name              = "/aws/api-gateway/customer-churn-api/access-logs"
  retention_in_days = 30

  tags = {
    Project   = "customer-churn-mlops"
    ManagedBy = "terraform"
  }
}

# -----------------------------------------------------
# REST API
# -----------------------------------------------------

resource "aws_api_gateway_rest_api" "churn_api" {
  name        = "customer-churn-api"
  description = "REST API for customer churn prediction inference"

  tags = {
    Project   = "customer-churn-mlops"
    ManagedBy = "terraform"
  }
}

# -----------------------------------------------------
# /predict resource
# -----------------------------------------------------

resource "aws_api_gateway_resource" "predict" {
  rest_api_id = aws_api_gateway_rest_api.churn_api.id
  parent_id   = aws_api_gateway_rest_api.churn_api.root_resource_id
  path_part   = "predict"
}

# -----------------------------------------------------
# POST /predict method
# -----------------------------------------------------

resource "aws_api_gateway_method" "predict_post" {
  rest_api_id   = aws_api_gateway_rest_api.churn_api.id
  resource_id   = aws_api_gateway_resource.predict.id
  http_method   = "POST"
  authorization = "NONE"
}

# -----------------------------------------------------
# Integration — forwards POST /predict to Lambda
# -----------------------------------------------------

resource "aws_api_gateway_integration" "predict_sagemaker" {
  rest_api_id             = aws_api_gateway_rest_api.churn_api.id
  resource_id             = aws_api_gateway_resource.predict.id
  http_method             = aws_api_gateway_method.predict_post.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.predict.invoke_arn
}

# -----------------------------------------------------
# Method response — 200 OK
# -----------------------------------------------------

resource "aws_api_gateway_method_response" "predict_200" {
  rest_api_id = aws_api_gateway_rest_api.churn_api.id
  resource_id = aws_api_gateway_resource.predict.id
  http_method = aws_api_gateway_method.predict_post.http_method
  status_code = "200"
}

# -----------------------------------------------------
# Integration response — passes SageMaker response back
# -----------------------------------------------------

resource "aws_api_gateway_integration_response" "predict_integration_response" {
  rest_api_id = aws_api_gateway_rest_api.churn_api.id
  resource_id = aws_api_gateway_resource.predict.id
  http_method = aws_api_gateway_method.predict_post.http_method
  status_code = aws_api_gateway_method_response.predict_200.status_code

  depends_on = [aws_api_gateway_integration.predict_sagemaker]
}

# -----------------------------------------------------
# Deployment and stage
# -----------------------------------------------------

resource "aws_api_gateway_deployment" "churn_api" {
  rest_api_id = aws_api_gateway_rest_api.churn_api.id

  # Force redeployment when the integration changes
  triggers = {
    redeployment = sha1(jsonencode([
      aws_api_gateway_resource.predict,
      aws_api_gateway_method.predict_post,
      aws_api_gateway_integration.predict_sagemaker,
      aws_lambda_function.predict.arn,
    ]))
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_api_gateway_stage" "prod" {
  rest_api_id   = aws_api_gateway_rest_api.churn_api.id
  deployment_id = aws_api_gateway_deployment.churn_api.id
  stage_name    = "prod"

  # Access logging — captures every request/response for data capture
  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.apigw_access_logs.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      resourcePath   = "$context.resourcePath"
      status         = "$context.status"
      responseLength = "$context.responseLength"
      ip             = "$context.identity.sourceIp"
      userAgent      = "$context.identity.userAgent"
      integrationLatency = "$context.integration.latency"
      responseLatency    = "$context.responseLatency"
      errorMessage       = "$context.error.message"
    })
  }

  depends_on = [aws_api_gateway_account.main]

  tags = {
    Project   = "customer-churn-mlops"
    ManagedBy = "terraform"
  }
}

# Method-level settings — enables CloudWatch metrics and execution logging
resource "aws_api_gateway_method_settings" "prod" {
  rest_api_id = aws_api_gateway_rest_api.churn_api.id
  stage_name  = aws_api_gateway_stage.prod.stage_name
  method_path = "*/*"

  settings {
    metrics_enabled        = true
    logging_level          = "INFO"
    data_trace_enabled     = true
    throttling_burst_limit = 100
    throttling_rate_limit  = 50
  }
}


# -----------------------------------------------------
# Outputs
# -----------------------------------------------------

output "api_gateway_invoke_url" {
  description = "URL to invoke the churn prediction API"
  value       = "${aws_api_gateway_stage.prod.invoke_url}/predict"
}
