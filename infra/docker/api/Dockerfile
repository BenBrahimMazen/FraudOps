# FastAPI scoring service (also used by the one-shot bootstrap/train jobs).
FROM python:3.11-slim

# libgomp1: OpenMP runtime required by LightGBM's compiled wheel
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.1 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Layer-cacheable: dependencies first, then the (small) source tree
COPY pyproject.toml uv.lock ./
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "fraudops.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
