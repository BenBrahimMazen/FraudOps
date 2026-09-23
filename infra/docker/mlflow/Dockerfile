# MLflow tracking server: Postgres backend store, MinIO (S3) artifacts.
FROM python:3.11-slim

# mlflow version pinned to match the application lock file (uv.lock)
RUN pip install --no-cache-dir "mlflow==3.16.1" psycopg2-binary boto3

EXPOSE 5000
