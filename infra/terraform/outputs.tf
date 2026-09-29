output "artifacts_bucket" {
  description = "S3 bucket backing the MLflow registry."
  value       = aws_s3_bucket.artifacts.id
}

output "ecr_repository_url" {
  description = "Push the serving image here (set enable_compute=true to plan this layer)."
  value       = var.enable_compute ? aws_ecr_repository.api[0].repository_url : "compute layer disabled (enable_compute=false)"
}

output "alb_dns_name" {
  description = "Score against this endpoint."
  value       = var.enable_compute ? aws_lb.api[0].dns_name : "compute layer disabled (enable_compute=false)"
}

output "ecs_service_name" {
  description = "Scale by changing desired_count on this service."
  value       = var.enable_compute ? aws_ecs_service.api[0].name : "compute layer disabled (enable_compute=false)"
}
