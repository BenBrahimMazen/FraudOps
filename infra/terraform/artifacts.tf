###############################################################################
# Artifact storage: one S3 bucket backs the MLflow model registry.
###############################################################################

resource "aws_s3_bucket" "artifacts" {
  bucket = var.artifacts_bucket_name
  tags   = local.common_tags
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled" # every promoted model version stays retrievable
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Artifacts are immutable per MLflow run; lifecycle just expires abandoned
# noncurrent versions of failed experiments after 90 days.
resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    id     = "expire-noncurrent"
    status = "Enabled"
    filter {} # all objects
    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}
