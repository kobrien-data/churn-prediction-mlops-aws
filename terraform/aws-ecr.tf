resource "aws_ecr_repository" "training" {
  name = "customer-churn-training"
  force_destroy= true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "inference" {
  name = "customer-churn-inference"
  force_destroy = true
  

  image_scanning_configuration {
    scan_on_push = true
  }
}