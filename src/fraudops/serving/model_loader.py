"""Champion model loading and zero-downtime reload.

The serving layer always scores the model behind the ``champion`` alias. A
background poller notices alias changes and swaps an atomically-held
reference; ``/admin/reload`` forces an immediate reload. Callers hold a
LoadedModel for the duration of one request, so an in-flight swap never
changes the model under a running request.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime

import mlflow
from mlflow.tracking import MlflowClient

from fraudops.registry import client as registry_client
from fraudops.registry.wrapper import FraudOpsModel


@dataclass(frozen=True)
class LoadedModel:
    version: int
    run_id: str
    bundle: FraudOpsModel
    loaded_at: datetime


class ModelHolder:
    """Thread-safe holder of the current champion."""

    def __init__(
        self,
        tracking_uri: str,
        model_name: str,
        poll_seconds: float = 15.0,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.poll_seconds = poll_seconds
        self._lock = threading.Lock()
        self._loaded: LoadedModel | None = None
        # models:/ alias URIs resolve through MLflow's process-global tracking
        # configuration — keep it pointed at our server for this holder.
        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient(tracking_uri=tracking_uri)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.reload_count = 0
        self.last_error: str | None = None

    # ------------------------------------------------------------------ load
    def load(self) -> LoadedModel:
        """Load the champion (raises when no champion alias is set)."""
        version = registry_client.version_for_name(
            self._client, self.model_name, registry_client.CHAMPION
        )
        if version is None:
            raise RuntimeError(f"no @{registry_client.CHAMPION} alias for {self.model_name}")
        pyfunc = mlflow.pyfunc.load_model(f"models:/{self.model_name}@{registry_client.CHAMPION}")
        loaded = LoadedModel(
            version=int(version),
            run_id=_run_id_for(self._client, self.model_name, version),
            bundle=FraudOpsModel.unwrap(pyfunc),
            loaded_at=datetime.now(UTC),
        )
        with self._lock:
            self._loaded = loaded
        self.reload_count += 1
        self.last_error = None
        return loaded

    def get(self) -> LoadedModel | None:
        with self._lock:
            return self._loaded

    def require(self) -> LoadedModel:
        current = self.get()
        if current is None:
            raise RuntimeError("no model loaded (champion alias unset or load failed)")
        return current

    # ------------------------------------------------------------- reload
    def reload_if_changed(self) -> bool:
        """Swap in a new champion when the alias points elsewhere."""
        try:
            current = self.get()
            version = registry_client.version_for_name(
                self._client, self.model_name, registry_client.CHAMPION
            )
            if version is None:
                return False
            if current is not None and int(version) == current.version:
                return False
            self.load()
            return True
        except Exception as exc:  # noqa: BLE001 — poller must never die
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

    # -------------------------------------------------------------- poller
    def start_polling(self) -> None:
        if self._thread is not None:
            return

        def _loop() -> None:
            while not self._stop.wait(self.poll_seconds):
                self.reload_if_changed()

        self._thread = threading.Thread(target=_loop, name="model-poller", daemon=True)
        self._thread.start()

    def stop_polling(self) -> None:
        self._stop.set()


def _run_id_for(client: MlflowClient, name: str, version: str) -> str:
    mv = client.get_model_version(name, version)
    if mv.run_id is None:  # registry rows always carry one; satisfy the stubs
        raise RuntimeError(f"model version {name}/{version} has no source run")
    return mv.run_id
