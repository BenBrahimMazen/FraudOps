###############################################################################
# IAM: two roles, least privilege each.
#   - execution role: what the ECS agent uses to pull the image and write logs
#   - task role: what the API process itself uses (read model artifacts only)
###############################################################################

data "aws_iam_policy_document" "assume_ecs_tasks" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${local.name_prefix}-api-execution"
  assume_role_policy = data.aws_iam_policy_document.assume_ecs_tasks.json
  tags               = local.common_tags
}

resource "aws_iam_role_policy_attachment" "execution_managed" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "task" {
  name               = "${local.name_prefix}-api-task"
  assume_role_policy = data.aws_iam_policy_document.assume_ecs_tasks.json
  tags               = local.common_tags
}

data "aws_iam_policy_document" "task_read_artifacts" {
  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:GetObjectVersion",
    ]
    resources = ["${aws_s3_bucket.artifacts.arn}/*"]
  }
  statement {
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.artifacts.arn]
  }
}

resource "aws_iam_role_policy" "task_read_artifacts" {
  name   = "read-model-artifacts"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task_read_artifacts.json
}
