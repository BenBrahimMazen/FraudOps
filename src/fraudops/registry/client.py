"""Champion/challenger alias helpers over the MLflow Model Registry.

Pure registry operations (register, alias, promote, rollback, status). The
promotion GATE (challenger must prove itself) arrives in the closed-loop
phase and will build on these primitives.
"""

from __future__ import annotations

import contextlib
import os
from typing import cast

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

CHALLENGER = "challenger"
CHAMPION = "champion"


def model_name() -> str:
    """Registered model name (env override for tests, default from config)."""
    return os.environ.get("FRAUDOPS_MODEL_NAME", "fraudops-lightgbm")


def configure(tracking_uri: str | None = None) -> MlflowClient:
    """Point MLflow at the tracking server (env var if no URI given)."""
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "./mlruns/mlflow.db")
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def _ensure_registered_model(client: MlflowClient, name: str) -> None:
    with contextlib.suppress(MlflowException):
        client.create_registered_model(name)  # exists already -> suppressed


def register_run_version(client: MlflowClient, run_id: str, model_uri: str | None = None) -> str:
    """Register the run's model as a new version; return the version number."""
    name = model_name()
    _ensure_registered_model(client, name)
    model_uri = model_uri or f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, name)
    return str(result.version)


def set_alias(client: MlflowClient, alias: str, version: str) -> None:
    client.set_registered_model_alias(model_name(), alias, str(version))


def version_for_name(client: MlflowClient, name: str, alias: str) -> str | None:
    """Version behind an alias for an explicit model name (None if unset)."""
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except MlflowException:
        return None


def get_version_by_alias(client: MlflowClient, alias: str) -> str | None:
    """Current version behind an alias, or None when the alias is unset."""
    return version_for_name(client, model_name(), alias)


def alias_run_id(client: MlflowClient, alias: str) -> str | None:
    try:
        mv = client.get_model_version_by_alias(model_name(), alias)
        return mv.run_id
    except MlflowException:
        return None


def promote_challenger(client: MlflowClient) -> dict:
    """Move the champion alias to the challenger's version (ungated promotion).

    The gated version (cost + PR-AUC checks with margin) replaces this in the
    closed-loop phase.
    """
    challenger = get_version_by_alias(client, CHALLENGER)
    if challenger is None:
        raise RuntimeError("no challenger alias set — train and register first")
    previous = get_version_by_alias(client, CHAMPION)
    set_alias(client, CHAMPION, challenger)
    return {"from": previous, "to": challenger}


def rollback_champion(client: MlflowClient, to_version: str | None = None) -> dict:
    """Restore an earlier champion version (latest-but-one when unspecified)."""
    versions = [int(mv.version) for mv in client.search_model_versions(f"name='{model_name()}'")]
    if not versions:
        raise RuntimeError("no registered versions to roll back to")
    current = get_version_by_alias(client, CHAMPION)
    if to_version is None:
        older = [v for v in sorted(versions, reverse=True) if str(v) != current]
        if not older:
            raise RuntimeError("no previous version to roll back to")
        to_version = str(older[0])
    set_alias(client, CHAMPION, str(to_version))
    return {"from": current, "to": str(to_version)}


def status(client: MlflowClient) -> dict:
    """Alias map plus each version's source run."""
    aliases = {alias: get_version_by_alias(client, alias) for alias in (CHAMPION, CHALLENGER)}
    versions = sorted(
        (
            {
                "version": int(mv.version),
                "run_id": mv.run_id,
                "registered_at": str(mv.creation_timestamp),
            }
            for mv in client.search_model_versions(f"name='{model_name()}'")
        ),
        key=lambda v: cast(int, v["version"]),  # version is int by construction
        reverse=True,
    )
    return {"model_name": model_name(), "aliases": aliases, "versions": versions[:10]}
