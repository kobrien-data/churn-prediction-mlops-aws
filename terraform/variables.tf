variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "eu-north-1"
}

variable "local_ip_addr" {
  description = "Your public IP address for SSH and MLflow access"
  type        = string
}