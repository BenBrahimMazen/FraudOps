"""Build-time step of the demo image: bake a local MLflow registry.

Creates a file-store registry at /app/mlruns containing one run carrying the
export metadata and one model version whose source is the /app/model
directory shipped in the image, with the ``champion`` alias set. The
unmodified serving app (ModelHolder) then boots with a registry-backed
champion and no external services: no tracking server, no object store, no
database.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

TRACKING_URI = "file:///app/mlruns"
MODEL_DIR = Path("/app/demo/model")
META_PATH = Path("/app/demo/meta.json")


def main() -> None:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    if not (MODEL_DIR / "MLmodel").exists():
        raise SystemExit(f"no model at {MODEL_DIR}; the export step must run first")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI, registry_uri=TRACKING_URI)

    experiment = client.get_experiment_by_name("demo-export")
    experiment_id = (
        experiment.experiment_id if experiment else client.create_experiment("demo-export")
    )
    run = client.create_run(experiment_id)
    for key, value in meta.items():
        client.log_param(run.info.run_id, key, str(value))
    client.set_terminated(run.info.run_id)

    with contextlib.suppress(MlflowException):  # exists on layer-cache rebuilds
        client.create_registered_model(meta["model_name"])
    # pad with placeholder versions so the baked one lands on the same number
    # as the registry it was exported from (/model and /score report it)
    for _ in range(int(meta["source_version"]) - 1):
        client.create_model_version(
            meta["model_name"], source=MODEL_DIR.as_uri(), run_id=run.info.run_id
        )
    version = client.create_model_version(
        meta["model_name"], source=MODEL_DIR.as_uri(), run_id=run.info.run_id
    )
    client.set_registered_model_alias(meta["model_name"], "champion", version.version)
    print(
        f"registered {meta['model_name']} v{version.version} "
        f"(source_version {meta['source_version']}) as @champion"
    )


if __name__ == "__main__":
    main()
