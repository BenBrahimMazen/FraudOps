"""FraudOps scoring API.

Contract (see README, API section):
- POST /score          one transaction -> probability, decision, threshold,
                        model version, top-3 SHAP reasons, latency
- POST /score/batch    up to 1000 transactions
- GET  /health         liveness + model status
- GET  /model          current champion details
- GET  /metrics        Prometheus exposition
- POST /admin/reload   force champion reload (X-Admin-Token header)

Every scored transaction is enqueued for the predictions store (buffered,
non-blocking). No champion available -> 503 with a clear error.
"""

from __future__ import annotations

import logging
import math
import os
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response

from fraudops.serving import metrics
from fraudops.serving.metrics import (
    fraud_score,
    http_request_latency_seconds,
    http_requests_total,
    model_info,
    scoring_decisions_total,
    scoring_latency_seconds,
)
from fraudops.serving.model_loader import ModelHolder
from fraudops.serving.persistence import Prediction, PredictionSink
from fraudops.serving.reasons import reasons_from_contributions
from fraudops.serving.schemas import BatchRequest, ScoreResponse, Transaction

logger = logging.getLogger("fraudops.serving")

ALERT = "alert"
APPROVE = "approve"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


class Service:
    """Wiring the endpoints use; overridable in tests."""

    def __init__(self, holder: ModelHolder, sink: PredictionSink, admin_token: str) -> None:
        self.holder = holder
        self.sink = sink
        self.admin_token = admin_token


def create_app(
    tracking_uri: str | None = None,
    model_name: str | None = None,
    database_dsn: str | None = None,
    admin_token: str | None = None,
) -> FastAPI:
    holder = ModelHolder(
        tracking_uri=tracking_uri or _env("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        model_name=model_name or _env("FRAUDOPS_MODEL_NAME", "fraudops-lightgbm"),
    )
    dsn = database_dsn if database_dsn is not None else _env("DATABASE_URL", "")
    sink = PredictionSink(dsn or None)
    service = Service(
        holder=holder,
        sink=sink,
        admin_token=admin_token or _env("FRAUDOPS_ADMIN_TOKEN", ""),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            loaded = holder.load()
            model_info.labels(version=str(loaded.version)).set(1)
            logger.info(
                "champion loaded: version %s (threshold %.4f)",
                loaded.version,
                loaded.bundle.threshold,
            )
        except Exception as exc:  # noqa: BLE001 — serve /health even without a model
            logger.warning("no champion available at startup: %s", exc)
        holder.start_polling()
        sink.start()
        yield
        holder.stop_polling()
        sink.stop()

    app = FastAPI(title="FraudOps scoring API", lifespan=lifespan)
    app.state.service = service

    def _require_model():
        try:
            return service.holder.require()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def _check_admin(x_admin_token: str | None = Header(default=None)) -> None:
        if not service.admin_token or x_admin_token != service.admin_token:
            raise HTTPException(status_code=401, detail="invalid admin token")

    def _record(loaded, tx, probability: float, decision: str, latency_ms: float) -> None:
        scoring_decisions_total.labels(decision=decision).inc()
        fraud_score.observe(probability)
        service.sink.submit(
            Prediction(
                transaction_id=tx.TransactionID,
                model_version=loaded.version,
                fraud_probability=probability,
                decision=decision,
                threshold=loaded.bundle.threshold,
                latency_ms=latency_ms,
                scored_at=time.time(),
            )
        )

    # ----------------------------------------------------------------- score
    @app.post("/score", response_model=ScoreResponse)
    def score(
        body: Transaction,
        loaded=Depends(_require_model),  # noqa: B008
    ) -> dict:
        t0 = time.perf_counter()
        df = pd.DataFrame([body.model_dump()])
        X = loaded.bundle.pipeline.transform(df)
        # one native TreeSHAP pass: contributions + base value; the raw margin
        # (sum) maps to the fraud probability through the logistic function
        contrib_row = loaded.bundle.model.booster_.predict(X, pred_contrib=True)[0]
        probability = float(1.0 / (1.0 + math.exp(-float(contrib_row.sum()))))
        decision = ALERT if probability >= loaded.bundle.threshold else APPROVE
        reasons = reasons_from_contributions(contrib_row[:-1], X.iloc[[0]])
        scoring_latency_seconds.observe(time.perf_counter() - t0)
        latency_ms = (time.perf_counter() - t0) * 1000

        _record(loaded, body, probability, decision, latency_ms)
        return {
            "fraud_probability": probability,
            "decision": decision,
            "threshold": loaded.bundle.threshold,
            "model_version": loaded.version,
            "top_reasons": reasons,
            "latency_ms": round(latency_ms, 3),
        }

    @app.post("/score/batch")
    def score_batch(
        body: BatchRequest,
        loaded=Depends(_require_model),  # noqa: B008
    ) -> dict:
        if not body.transactions:
            return {
                "model_version": loaded.version,
                "threshold": loaded.bundle.threshold,
                "latency_ms": 0.0,
                "results": [],
            }
        t0 = time.perf_counter()
        df = pd.DataFrame([t.model_dump() for t in body.transactions])
        _, probs = loaded.bundle.score_with_features(df)
        probabilities = [float(p) for p in probs]
        decisions = [ALERT if p >= loaded.bundle.threshold else APPROVE for p in probabilities]
        scoring_latency_seconds.observe(time.perf_counter() - t0)
        latency_ms = (time.perf_counter() - t0) * 1000
        for tx, probability, decision in zip(
            body.transactions, probabilities, decisions, strict=True
        ):
            _record(loaded, tx, probability, decision, latency_ms)
        return {
            "model_version": loaded.version,
            "threshold": loaded.bundle.threshold,
            "latency_ms": round(latency_ms, 3),
            "results": [
                {"fraud_probability": p, "decision": d}
                for p, d in zip(probabilities, decisions, strict=True)
            ],
        }

    # ------------------------------------------------------------------- ops
    @app.get("/health")
    def health() -> dict:
        loaded = service.holder.get()
        return {
            "status": "ok",
            "model_loaded": loaded is not None,
            "model_version": loaded.version if loaded else None,
        }

    @app.get("/model")
    def model() -> dict:
        loaded = _require_model()
        meta = loaded.bundle.metadata
        return {
            "model_name": service.holder.model_name,
            "version": loaded.version,
            "run_id": loaded.run_id,
            "threshold": loaded.bundle.threshold,
            "trained_at": meta.get("trained_at"),
            "train_window": {
                "train_days": meta.get("train_days"),
                "val_days": meta.get("val_days"),
            },
            "metrics": meta.get("metrics"),
            "loaded_at": loaded.loaded_at.isoformat(),
            "reload_count": service.holder.reload_count,
        }

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        payload, content_type = metrics.render()
        return Response(content=payload, media_type=content_type)

    @app.post("/admin/reload")
    def reload_model(_: None = Depends(_check_admin)) -> dict:
        try:
            loaded = service.holder.load()
        except Exception as exc:  # noqa: BLE001 — surface as 503, not a crash
            raise HTTPException(status_code=503, detail=f"reload failed: {exc}") from exc
        model_info.labels(version=str(loaded.version)).set(1)
        return {"model_version": loaded.version, "run_id": loaded.run_id}

    @app.middleware("http")
    async def _instrument(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        path = request.url.path
        http_requests_total.labels(path=path, code=str(response.status_code)).inc()
        http_request_latency_seconds.labels(path=path).observe(elapsed)
        return response

    return app


app = create_app()
