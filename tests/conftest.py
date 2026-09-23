"""Shared fixtures: a tiny real model registered in a throwaway MLflow store."""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from fraudops.features.pipeline import FeaturePipeline
from fraudops.registry import client as registry_client
from fraudops.registry.wrapper import FraudOpsModel

MODEL_NAME = "test-fraudops-model"


def make_training_frame(seed: int = 0, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    is_fraud = (signal + rng.normal(scale=0.7, size=n) > 1.6).astype(int)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": rng.integers(0, 30 * 86_400, size=n),
            "TransactionAmt": rng.uniform(1, 1000, size=n),
            "ProductCD": rng.choice(["W", "H", "C"], size=n),
            "card4": rng.choice(["visa", "mastercard"], size=n),
            "C1": signal,
            "D1": rng.uniform(0, 30, size=n),
            "id_02": rng.uniform(0, 50, size=n),
            "DeviceInfo": rng.choice(["Windows", "iOS"], size=n),
            "has_identity": rng.integers(0, 2, size=n).astype(bool),
            "isFraud": is_fraud,
        }
    )


@pytest.fixture()
def trained_bundle():
    """Tiny real LightGBM + fitted pipeline + threshold (fast, deterministic)."""
    from lightgbm import LGBMClassifier

    frame = make_training_frame()
    pipeline = FeaturePipeline().fit(frame)
    X = pipeline.transform(frame)
    model = LGBMClassifier(
        n_estimators=25, num_leaves=7, learning_rate=0.2, random_state=0, verbosity=-1
    )
    model.fit(X, frame["isFraud"])
    scores = model.predict_proba(X)[:, 1]
    # trivial cost-optimal threshold on the training frame (test-only model)
    from fraudops.models.threshold import optimal_cost_threshold

    threshold = optimal_cost_threshold(
        frame["isFraud"].to_numpy(),
        frame["TransactionAmt"].to_numpy(),
        scores,
        false_alert_cost=5.0,
    )
    metadata = {
        "threshold": float(threshold),
        "train_days": 20,
        "val_days": 5,
        "trained_at": "2026-09-22T00:00:00+00:00",
        "metrics": {"validation": {"pr_auc": 0.9, "cost_per_100k": 1234.0}},
    }
    return FraudOpsModel(model, pipeline, float(threshold), metadata), frame


@pytest.fixture()
def registry(tmp_path: Path, trained_bundle, monkeypatch) -> dict:
    """SQLite-backed MLflow tracking + registry with the model as champion."""
    store = tmp_path / "mlruns"
    store.mkdir()
    uri = f"sqlite:///{(store / 'test.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("FRAUDOPS_MODEL_NAME", MODEL_NAME)
    monkeypatch.delenv("MLFLOW_S3_ENDPOINT_URL", raising=False)

    bundle, _ = trained_bundle
    mlflow.set_tracking_uri(uri)
    exp = mlflow.set_experiment("fraudops_training")
    with mlflow.start_run(experiment_id=exp.experiment_id) as run:
        mlflow.pyfunc.log_model(name="model", python_model=bundle)
        run_id = run.info.run_id
    cli = registry_client.configure(uri)
    version = registry_client.register_run_version(cli, run_id)
    registry_client.set_alias(cli, registry_client.CHAMPION, version)
    return {"uri": uri, "run_id": run_id, "version": int(version), "client": cli}


@pytest.fixture()
def sample_transactions() -> list[dict]:
    frame = make_training_frame(seed=99, n=3)
    rows = frame.drop(columns=["isFraud"]).to_dict(orient="records")
    # numpy scalars are not JSON-serializable: convert to plain Python types
    converted = []
    for row in rows:
        converted.append(
            {
                key: int(v)
                if isinstance(v, np.integer)
                else float(v)
                if isinstance(v, np.floating)
                else v
                for key, v in row.items()
            }
        )
    return converted


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Keep host-side env (docker stack URLs) out of unit tests."""
    for var in ("DATABASE_URL", "FRAUDOPS_ADMIN_TOKEN", "MLFLOW_S3_ENDPOINT_URL"):
        monkeypatch.delenv(var, raising=False)
    yield
