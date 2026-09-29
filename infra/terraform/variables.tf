variable "region" {
  description = "AWS region for every resource (any region works against LocalStack)."
  type        = string
  default     = "eu-west-1"
}

variable "localstack_endpoint" {
  description = "Base URL of the LocalStack edge. Inside the compose network: http://localstack:4566."
  type        = string
  default     = "http://localstack:4566"
}

variable "artifacts_bucket_name" {
  description = "S3 bucket for MLflow model artifacts."
  type        = string
  default     = "fraudops-mlflow-artifacts"
}

variable "api_image_tag" {
  description = "Tag of the serving image pushed to ECR."
  type        = string
  default     = "localstack-demo"
}

variable "api_desired_count" {
  description = "Fargate task count for the serving service."
  type        = number
  default     = 2
}

variable "api_environment" {
  description = "Environment for the serving container; endpoints for managed services (MLflow, Postgres, Kafka) are injected here. In a real account the token would come from SSM, not a tf file."
  type        = map(string)
  default = {
    MLFLOW_TRACKING_URI    = "http://mlflow.internal:5000"
    MLFLOW_S3_ENDPOINT_URL = "" # empty on real AWS; only the local stack needs it
    FRAUDOPS_MODEL_NAME    = "fraudops-lightgbm"
    FRAUDOPS_ADMIN_TOKEN   = "change-me-via-ssm"
  }
}

variable "enable_compute" {
  description = "Plan the ECR/ECS/ALB serving layer. Default false: LocalStack Community (the zero-cost license) does not implement ECS or ELBv2, and ECR rejects static test credentials, so `make tf-apply` covers the storage/IAM/logging layer only. The compute layer is verified by `terraform validate` in CI and by `terraform plan -var enable_compute=true`."
  type        = bool
  default     = false
}
