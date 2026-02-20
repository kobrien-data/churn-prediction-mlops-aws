# -----------------------------------------------------
# SageMaker Domain
# This is the top-level container for SageMaker Studio.
# Think of it as the workspace — everything else
# (users, apps, notebooks) lives inside it
# -----------------------------------------------------

resource "aws_sagemaker_domain" "main" {
  domain_name = "customer-churn-domain"

  # IAM Identity Center is the modern auth approach but
  # requires additional setup. IAM mode is simpler and
  # sufficient for a single-user portfolio project
  auth_mode = "IAM"

  vpc_id     = data.aws_vpc.default.id
  subnet_ids = data.aws_subnets.default.ids

  default_user_settings {
    # The execution role we created in iam.tf
    # This is what gives Studio users permission to
    # access S3, run training jobs, etc.
    execution_role = aws_iam_role.sagemaker_execution_role.arn

    # Controls how Studio is accessed
    # JupyterLab gives you the modern Studio interface
    jupyter_server_app_settings {
      default_resource_spec {
        instance_type = "system"
        # "system" means no dedicated instance for the
        # Studio UI itself — it only spins up compute
        # when you actually run something, keeping
        # costs at zero when idle
      }
    }

    # Kernel gateway is what powers the actual notebook
    # kernels when you run code in Studio
    kernel_gateway_app_settings {
      default_resource_spec {
        instance_type = "ml.t3.medium"
        # t3.medium is the smallest paid instance type
        # Only use this for interactive development —
        # training jobs run on their own separate
        # instances defined in your pipeline config
      }
    }
  }

  # Ensures the domain is fully deleted when you run
  # terraform destroy, including any apps running inside
  # it. Without this, destroy will fail if Studio is open
  retention_policy {
    home_efs_file_system = "Delete"
  }

  tags = {
    Project   = "customer-churn-mlops"
    ManagedBy = "terraform"
  }
}

# -----------------------------------------------------
# SageMaker User Profile
# A user profile is an individual user inside the domain
# For a solo portfolio project you only need one, but
# in a team setting each person would have their own
# profile with their own settings and home directory
# -----------------------------------------------------

resource "aws_sagemaker_user_profile" "main" {
  domain_id         = aws_sagemaker_domain.main.id
  user_profile_name = "default-user"

  user_settings {
    # Inherits the execution role from the domain
    # You could override this per-user if different
    # users needed different permissions, but for a
    # solo project this is fine
    execution_role = aws_iam_role.sagemaker_execution_role.arn
  }

  tags = {
    Project   = "customer-churn-mlops"
    ManagedBy = "terraform"
  }
}

# -----------------------------------------------------
# Outputs
# Expose key values for reference in other files
# or for use with boto3 scripts
# -----------------------------------------------------

output "sagemaker_domain_id" {
  description = "SageMaker Domain ID — needed for boto3 pipeline scripts"
  value       = aws_sagemaker_domain.main.id
}

output "sagemaker_domain_url" {
  description = "URL to access SageMaker Studio in the console"
  value       = aws_sagemaker_domain.main.url
}

output "sagemaker_user_profile_name" {
  description = "User profile name — needed for boto3 scripts"
  value       = aws_sagemaker_user_profile.main.user_profile_name
}
