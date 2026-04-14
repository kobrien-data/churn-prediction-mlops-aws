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

  tags = {
    Project   = "customer-churn-mlops"
    ManagedBy = "terraform"
  }
}


# -----------------------------------------------------
# Outputs
# -----------------------------------------------------

output "api_gateway_invoke_url" {
  description = "URL to invoke the churn prediction API"
  value       = "${aws_api_gateway_stage.prod.invoke_url}/predict"
}
