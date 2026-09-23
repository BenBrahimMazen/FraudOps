# Airflow scheduler + API server with the fraudops package installed.
FROM apache/airflow:3.0.4-python3.11
USER root
COPY pyproject.toml uv.lock /tmp/fraudops/
COPY src /tmp/fraudops/src
COPY README.md /tmp/fraudops/
RUN chown -R airflow: /tmp/fraudops
USER airflow
RUN pip install --no-cache-dir /tmp/fraudops psycopg2-binary
