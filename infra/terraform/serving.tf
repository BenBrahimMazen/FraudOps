###############################################################################
# Serving: ECR image + ECS Fargate service behind an ALB.
#
# Two layers, because of what zero-cost LocalStack can actually emulate:
#   - applied by `make tf-apply` (Community license): CloudWatch log group,
#     security groups, VPC/subnet lookups — these emulate fine.
#   - gated behind `enable_compute` (default false, validate/plan only):
#     ECR, ECS, ELBv2. LocalStack Community does not implement ECS or ELBv2
#     (both need the paid Pro license) and its ECR rejects the static test
#     credentials, so this layer is verified by `terraform validate` in CI
#     and by `terraform plan -var enable_compute=true`.
#
# Task size note: one API worker saturates around 15 req/s on a laptop core
# (see README Phase 5), so horizontal task count is the scaling lever —
# desired_count defaults to 2.
###############################################################################

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.name_prefix}-api"
  retention_in_days = 30
  tags              = local.common_tags
}

# ------------------------------------------------------------------ networking
resource "aws_security_group" "alb" {
  name   = "${local.name_prefix}-alb"
  vpc_id = data.aws_vpc.default.id
  ingress {
    description = "public scoring traffic"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = local.common_tags
}

resource "aws_security_group" "api" {
  name   = "${local.name_prefix}-api"
  vpc_id = data.aws_vpc.default.id
  ingress {
    description     = "scoring traffic from the ALB only"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = local.common_tags
}

# The default VPC keeps this a demo-shaped stack; a real deployment would
# reference private subnets by ID (out of scope, noted in the README).
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  # NB: the idiomatic `default-vpc = true` filter is not implemented in
  # LocalStack's Moto backend; filtering by the VPC id is equivalent.
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ------------------------------------------------------------------ compute
# Everything below is behind enable_compute; see the header comment.
locals {
  compute = var.enable_compute ? 1 : 0
}

resource "aws_ecr_repository" "api" {
  count = local.compute
  name  = "${local.name_prefix}-api"
  image_scanning_configuration {
    scan_on_push = true
  }
  tags = local.common_tags
}

resource "aws_ecs_cluster" "main" {
  count = local.compute
  name  = local.name_prefix
  tags  = local.common_tags
}

resource "aws_ecs_task_definition" "api" {
  count                    = local.compute
  family                   = "${local.name_prefix}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 1024
  memory                   = 2048
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  tags                     = local.common_tags

  container_definitions = jsonencode([
    {
      name      = "api"
      image     = "${aws_ecr_repository.api[0].repository_url}:${var.api_image_tag}"
      essential = true
      portMappings = [
        { containerPort = 8000, protocol = "tcp" }
      ]
      environment = [for k, v in var.api_environment : { name = k, value = v }]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.api.name
          awslogs-region        = var.region
          awslogs-stream-prefix = "api"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])
}

resource "aws_lb" "api" {
  count              = local.compute
  name               = "${local.name_prefix}-api"
  load_balancer_type = "application"
  subnets            = data.aws_subnets.default.ids
  security_groups    = [aws_security_group.alb.id]
  tags               = local.common_tags
}

resource "aws_lb_target_group" "api" {
  count       = local.compute
  name        = "${local.name_prefix}-api"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip" # Fargate
  health_check {
    path                = "/health"
    interval            = 30
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
  tags = local.common_tags
}

resource "aws_lb_listener" "http" {
  count             = local.compute
  load_balancer_arn = aws_lb.api[0].arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api[0].arn
  }
}

resource "aws_ecs_service" "api" {
  count           = local.compute
  name            = "${local.name_prefix}-api"
  cluster         = aws_ecs_cluster.main[0].id
  task_definition = aws_ecs_task_definition.api[0].arn
  desired_count   = var.api_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.api.id]
    assign_public_ip = true # demo on the default VPC; private subnets + NAT at real scale
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api[0].arn
    container_name   = "api"
    container_port   = 8000
  }

  # zero-downtime deploys: let a new task pass health checks before draining
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200

  depends_on = [aws_lb_listener.http]
}
