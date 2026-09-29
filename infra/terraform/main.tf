###############################################################################
# FraudOps — plausible AWS deployment (validated against LocalStack ONLY)
#
# What this describes:
#   - S3 bucket for MLflow model artifacts            [applied to LocalStack]
#   - IAM task-execution and task roles              [applied to LocalStack]
#   - CloudWatch log group, security groups          [applied to LocalStack]
#   - ECR repository for the serving image           [validate/plan only]
#   - ECS Fargate service behind an ALB              [validate/plan only]
#
# The compute layer is gated behind `enable_compute` because LocalStack
# Community (zero-cost) does not implement ECS or ELBv2; see serving.tf.
#
# What this deliberately does NOT describe (managed services injected as env):
#   RDS Postgres, MSK Kafka, the MLflow tracking server, Grafana Cloud.
# Those endpoints are passed through `var.api_environment`.
#
# NEVER applied to real AWS. The provider is pinned to a LocalStack endpoint
# variable; the apply evidence in the README comes from `make tf-apply`.
###############################################################################

terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region

  # LocalStack test credentials; every skip flag matters — without them the
  # provider phones real AWS STS/metadata endpoints and fails.
  access_key                  = "test"
  secret_key                  = "test"
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true
  skip_region_validation      = true
  s3_use_path_style           = true # LocalStack needs path-style S3

  endpoints {
    s3    = var.localstack_endpoint
    ec2   = var.localstack_endpoint
    ecs   = var.localstack_endpoint
    iam   = var.localstack_endpoint
    logs  = var.localstack_endpoint
    elbv2 = var.localstack_endpoint
  }
}

locals {
  name_prefix = "fraudops"
  common_tags = {
    Project     = "fraudops"
    Environment = "localstack-demo"
    ManagedBy   = "terraform"
  }
}
