module "s3_raw_data" {
  source  = "terraform-aws-modules/s3-bucket/aws"
  version = "~> 4.0"

  bucket = "customer-churn-raw-data-941377133770"
  force_destroy = true

  control_object_ownership = true
  object_ownership = "BucketOwnerEnforced"

  versioning = {
    enabled = true
  }
}

module "s3_processed_data" {
  source  = "terraform-aws-modules/s3-bucket/aws"
  version = "~> 4.0"

  bucket = "customer-churn-processed-data"
  force_destroy = true

  control_object_ownership = true
  object_ownership = "BucketOwnerEnforced"

  versioning = {
    enabled = true
  }
}

module "s3_model_artifacts" {
  source  = "terraform-aws-modules/s3-bucket/aws"
  version = "~> 4.0"

  bucket = "customer-churn-model-artifacts"
  force_destroy = true

  control_object_ownership = true
  object_ownership = "BucketOwnerEnforced"

  versioning = {
    enabled = true
  }
}