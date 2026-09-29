"""Host-side deploy: stage the Space layout and upload it to Hugging Face.

Reads HF_TOKEN (a write token) and SPACE_ID (e.g. "your-user/fraudops")
from the environment, stages exactly the files the demo Dockerfile expects,
creates the Docker Space if needed (private, CPU basic tier - no card) and
uploads the staging folder. The Space builds and runs the image itself.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGING = REPO_ROOT / "demo" / "staging"

# The Space's README.md doubles as its metadata file (title, sdk, app_port).
# Written from this template so the live URL in the examples is real.
SPACE_README = """---
title: FraudOps
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

Real-time fraud scoring API (LightGBM + SHAP reason codes) — a
self-contained slice of the serving layer of
[FraudOps](https://github.com/BenBrahimMazen/FraudOps): the same FastAPI
application, with the champion model baked into the image. No backing
services, no state — every request is scored by the promoted model at its
cost-optimal threshold.

Interactive docs: **`/docs`** (Swagger UI).

Example — only three fields are required; anything omitted the model
treats as missing (which is itself informative):

    curl -X POST {base_url}/score \\
        -H "Content-Type: application/json" \\
        -d '{{"TransactionID": 3677001, "TransactionDT": 13200000, "TransactionAmt": 149.0}}'
    # -> fraud_probability 0.414, decision "alert"

    curl -X POST {base_url}/score \\
        -H "Content-Type: application/json" \\
        -d '{{"TransactionID": 3677005, "TransactionDT": 1330000, "TransactionAmt": 29.0,
             "ProductCD": "W", "card1": 10035, "card4": "discover", "card6": "credit",
             "C1": 6.0, "C2": 6.0, "C5": 5.0, "C13": 8.0, "C14": 6.0, "D1": 30.0}}'
    # -> fraud_probability 0.005, decision "approve"

Response: `fraud_probability`, `decision` (`alert`/`approve`), `threshold`,
`model_version`, top-3 SHAP `top_reasons` (feature, value, signed log-odds
contribution), `latency_ms`. The threshold is cost-optimal (missed fraud
costs the transaction amount, a false alert costs a fixed review) — which
is why it sits far below 0.5.

Also: `GET /model` (current champion, threshold, training window, metrics)
and `GET /metrics` (Prometheus exposition). Batch scoring up to 1,000
transactions via `POST /score/batch`.
"""


def stage(base_url: str) -> Path:
    if not (REPO_ROOT / "demo" / "model" / "MLmodel").exists():
        sys.exit("no exported champion under demo/model/ - run `make demo-export` first")
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "demo" / "Dockerfile", STAGING / "Dockerfile")
    for item in ("pyproject.toml", "uv.lock", "README.md"):
        shutil.copy2(REPO_ROOT / item, STAGING / item)
    # overwrite the repo README with the Space's own (front-matter + examples)
    (STAGING / "README.md").write_text(SPACE_README.format(base_url=base_url), encoding="utf-8")
    shutil.copytree(REPO_ROOT / "src", STAGING / "src")
    shutil.copytree(REPO_ROOT / "demo" / "model", STAGING / "demo" / "model")
    for item in ("register.py", "meta.json"):
        shutil.copy2(REPO_ROOT / "demo" / item, STAGING / "demo" / item)
    return STAGING


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    space_id = os.environ.get("SPACE_ID")
    if not token or not space_id:
        sys.exit("set HF_TOKEN (a write token) and SPACE_ID (e.g. your-user/fraudops)")

    base_url = f"https://{space_id}.hf.space"
    staging = stage(base_url)
    api = HfApi(token=token)
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="docker",
        private=True,
        exist_ok=True,
    )
    url = api.upload_folder(
        folder_path=str(staging),
        repo_id=space_id,
        repo_type="space",
        commit_message="deploy fraudops serving demo",
    )
    print(f"uploaded {space_id} -> {url} (app: {base_url})")


if __name__ == "__main__":
    main()
