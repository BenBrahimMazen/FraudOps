# Airflow scheduler + API server with the fraudops package installed.
#
# The package is installed under Airflow's official constraints so that
# shared dependencies (fastapi via mlflow, pandas, ...) resolve to versions
# Airflow 3.0.4 is tested with — the application lock file pins newer ones.
FROM apache/airflow:3.0.4-python3.11
USER root
COPY pyproject.toml uv.lock /tmp/fraudops/
COPY src /tmp/fraudops/src
COPY README.md /tmp/fraudops/
RUN chown -R airflow: /tmp/fraudops
USER airflow
RUN pip install --no-cache-dir \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.4/constraints-3.11.txt" \
    /tmp/fraudops psycopg2-binary
