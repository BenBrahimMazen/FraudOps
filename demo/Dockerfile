# Self-contained slice of the serving layer for the live demo: the real
# FastAPI app with a champion model baked into the image. No Postgres, no
# MinIO, no Kafka — scoring only (probability, decision, SHAP reasons).
#
# Build context is the repository root (make demo-build) or a Space repo
# with the same layout: pyproject.toml, uv.lock, README.md, src/, demo/.
FROM python:3.11-slim

# libgomp1: OpenMP runtime required by LightGBM's compiled wheel
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.1 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    MLFLOW_ALLOW_FILE_STORE=true

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# exported champion + the build-time step that re-registers it into a local
# MLflow file store, so the app finds a @champion alias at boot
COPY demo ./demo
RUN python demo/register.py

# the app itself, pointed at the baked-in file-store registry; persistence
# stays disabled (no DATABASE_URL) and admin reload stays locked (no token)
ENV MLFLOW_TRACKING_URI=file:///app/mlruns \
    FRAUDOPS_MODEL_NAME=fraudops-lightgbm

EXPOSE 7860
CMD ["uvicorn", "fraudops.serving.app:app", "--host", "0.0.0.0", "--port", "7860"]
