# selects AMI for ec2 instance
data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_key_pair" "mlflow" {
  key_name   = "churn-mlflow-key"
  public_key = file("~/.ssh/id_rsa.pub")
}

resource "aws_security_group" "mlflow" {
  name        = "churn-mlflow-sg"
  description = "Security group for MLflow server"

  # Allow MLflow UI access from your IP only
  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["${var.local_ip_addr}/32"]
  }

  # Allow SSH from your IP only
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["${var.local_ip_addr}/32"]
  }

  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Project   = "customer-churn-mlops"
    ManagedBy = "terraform"
  }
}

resource "aws_iam_instance_profile" "mlflow" {
  name = "churn-mlflow-instance-profile"
  role = aws_iam_role.mlflow_ec2.name
}

resource "aws_iam_role" "mlflow_ec2" {
  name = "churn-mlflow-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "mlflow_s3" {
  role       = aws_iam_role.mlflow_ec2.name
  policy_arn = aws_iam_policy.sagemaker_s3.arn
}

resource "aws_instance" "mlflow_instance" {
  ami           = data.aws_ami.amazon_linux_2023.id
  instance_type = "t3.micro"
  key_name               = aws_key_pair.mlflow.key_name
  vpc_security_group_ids = [aws_security_group.mlflow.id]
  iam_instance_profile   = aws_iam_instance_profile.mlflow.name

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  user_data = <<-EOF
    #!/bin/bash
    dnf update -y
    dnf install -y python3-pip
    pip3 install mlflow boto3 --ignore-installed
    mkdir -p /mlflow/artifacts

    # Start MLflow server
    mlflow server \
      --host 0.0.0.0 \
      --port 5000 \
      --default-artifact-root s3://customer-churn-model-artifacts-941377133770/mlflow \
      --backend-store-uri sqlite:////mlflow/mlflow.db &
EOF


  tags = {
    Name = "customer-churn-mlops"
  }
}