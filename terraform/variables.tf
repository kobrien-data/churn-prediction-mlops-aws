variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "eu-north-1"
}

variable "local_ip_addr" {
  description = "Your public IP address for SSH and MLflow access"
  type        = string
}

variable "sagemaker_endpoint_name" {
  description = "Name of the SageMaker inference endpoint"
  type        = string
  default     = "customer-churn-endpoint"
}