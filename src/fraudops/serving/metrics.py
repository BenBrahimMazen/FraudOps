"""Prometheus instrumentation for the scoring service."""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

registry = CollectorRegistry()

http_requests_total = Counter(
    "fraudops_http_requests_total",
    "HTTP requests handled",
    labelnames=["path", "code"],
    registry=registry,
)
http_request_latency_seconds = Histogram(
    "fraudops_http_request_latency_seconds",
    "HTTP request latency",
    labelnames=["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=registry,
)
scoring_latency_seconds = Histogram(
    "fraudops_scoring_latency_seconds",
    "Model scoring latency (features + predict + reasons)",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
    registry=registry,
)
scoring_decisions_total = Counter(
    "fraudops_scoring_decisions_total",
    "Decisions taken",
    labelnames=["decision"],
    registry=registry,
)
fraud_score = Histogram(
    "fraudops_score_distribution",
    "Distribution of fraud probabilities",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99),
    registry=registry,
)
model_reload_total = Counter(
    "fraudops_model_reload_total",
    "Champion model (re)loads",
    registry=registry,
)
model_info = Gauge(
    "fraudops_model_info",
    "Current champion (labels: version)",
    labelnames=["version"],
    registry=registry,
)
predictions_persisted_total = Counter(
    "fraudops_predictions_persisted_total",
    "Predictions written to the store",
    registry=registry,
)
predictions_dropped_total = Counter(
    "fraudops_predictions_dropped_total",
    "Predictions dropped (store unavailable / queue full)",
    registry=registry,
)


def render() -> tuple[bytes, str]:
    return generate_latest(registry), CONTENT_TYPE_LATEST
