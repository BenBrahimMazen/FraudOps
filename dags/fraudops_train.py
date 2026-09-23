"""Training pipeline: train, evaluate, register as challenger.

Manual trigger (or via the retrain-on-drift DAG in the closed-loop phase).
The two-step split keeps run/registration boundaries visible in the Airflow
UI while reusing the exact same library the CLI bootstrap uses.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

from airflow.decorators import dag, task

DATA_DIR = os.environ.get("FRAUDOPS_DATA_DIR", "/data/raw")


@dag(
    dag_id="fraudops_train",
    schedule=None,
    start_date=datetime(2026, 9, 22, tzinfo=UTC),
    catchup=False,
    tags=["fraudops", "training"],
)
def fraudops_train():
    @task
    def train_and_evaluate() -> dict:
        from fraudops.models.train import train_baseline

        out = train_baseline(data_dir=DATA_DIR)
        return {
            "run_id": out["run_id"],
            "threshold": out["threshold"],
            "validation": {k: round(v, 6) for k, v in out["metrics"]["validation"].items()},
        }

    @task
    def register_challenger(info: dict) -> int:
        from fraudops.registry import client as rc

        cli = rc.configure(os.environ["MLFLOW_TRACKING_URI"])
        version = rc.register_run_version(cli, info["run_id"])
        rc.set_alias(cli, rc.CHALLENGER, version)
        print(
            f"registered version {version} as challenger "
            f"(validation pr_auc={info['validation']['pr_auc']}, "
            f"threshold={info['threshold']:.4f})"
        )
        return int(version)

    register_challenger(train_and_evaluate())


fraudops_train()
