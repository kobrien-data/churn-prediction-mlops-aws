# -----------------------------------------------------
# Lambda deployment package
# -----------------------------------------------------

data "archive_file" "lambda" {
  type        = "zip"
  source_file = "${path.module}/../src/inference/lambda_handler.py"
  output_path = "${path.module}/../src/inference/lambda_handler.zip"
}

# -----------------------------------------------------
# Lambda function
# -----------------------------------------------------

resource "aws_lambda_function" "predict" {
  function_name    = "customer-churn-predict"
  filename         = data.archive_file.lambda.output_path
  source_code_hash = data.archive_file.lambda.output_base64sha256
  handler          = "lambda_handler.handler"
  runtime          = "python3.12"
  role             = aws_iam_role.lambda_execution_role.arn
  timeout          = 30

  environment {
    variables = {
      SAGEMAKER_ENDPOINT_NAME = var.sagemaker_endpoint_name
    }
  }

  tags = {
    Project   = "customer-churn-mlops"
    ManagedBy = "terraform"
  }
}

# -----------------------------------------------------
# IAM role for Lambda
# -----------------------------------------------------

resource "aws_iam_role" "lambda_execution_role" {
  name = "customer-churn-lambda-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = {
    Project   = "customer-churn-mlops"
    ManagedBy = "terraform"
  }
}

resource "aws_iam_role_policy" "lambda_sagemaker_policy" {
  name = "customer-churn-lambda-sagemaker-policy"
  role = aws_iam_role.lambda_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "sagemaker:InvokeEndpoint"
        Resource = "arn:aws:sagemaker:${var.aws_region}:*:endpoint/${var.sagemaker_endpoint_name}"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:log-group:/aws/lambda/customer-churn-*"
      }
    ]
  })
}

# -----------------------------------------------------
# Allow API Gateway to invoke Lambda
# -----------------------------------------------------

resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.predict.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.churn_api.execution_arn}/*/*"
}

# -----------------------------------------------------
# Outputs
# -----------------------------------------------------

output "lambda_function_name" {
  description = "Name of the predict Lambda function"
  value       = aws_lambda_function.predict.function_name
}
