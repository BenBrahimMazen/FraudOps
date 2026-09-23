"""API contract tests: valid/invalid input, no-model, batch limits, admin."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraudops.registry.wrapper import FraudOpsModel
from fraudops.serving.app import create_app
from fraudops.serving.persistence import Prediction, PredictionSink
from tests.conftest import MODEL_NAME, make_training_frame

ALERT = "alert"
APPROVE = "approve"


class CapturingSink(PredictionSink):
    """In-memory sink: no database, records everything for assertions."""

    def __init__(self) -> None:
        super().__init__(dsn=None)
        self.captured: list[Prediction] = []

    def submit(self, prediction: Prediction) -> None:
        self.captured.append(prediction)


@pytest.fixture()
def client(registry) -> TestClient:
    sink = CapturingSink()
    app = create_app(
        tracking_uri=registry["uri"],
        model_name=MODEL_NAME,
        database_dsn=None,
        admin_token="test-token",
    )
    app.state.service.sink = sink  # type: ignore[attr-defined]
    with TestClient(app) as tc:
        yield tc


def valid_transaction() -> dict:
    row = make_training_frame(seed=7, n=1).iloc[0].to_dict()
    out: dict = {}
    for key, value in row.items():
        if key == "isFraud":
            continue
        if isinstance(value, np.integer):
            out[key] = int(value)
        elif isinstance(value, np.floating):
            out[key] = float(value)
        else:
            out[key] = value
    return out


class TestScore:
    def test_valid_input_full_contract(self, client: TestClient) -> None:
        response = client.post("/score", json=valid_transaction())
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["decision"] in (ALERT, APPROVE)
        assert 0.0 <= body["threshold"] <= 1.0
        assert body["model_version"] >= 1
        assert body["latency_ms"] >= 0
        reasons = body["top_reasons"]
        assert len(reasons) == 3
        for reason in reasons:
            assert set(reason) == {"feature", "value", "contribution"}
            assert isinstance(reason["contribution"], float)

    def test_decision_matches_threshold_rule(self, client: TestClient) -> None:
        body = client.post("/score", json=valid_transaction()).json()
        expected = ALERT if body["fraud_probability"] >= body["threshold"] else APPROVE
        assert body["decision"] == expected

    def test_missing_required_field_rejected(self, client: TestClient) -> None:
        tx = valid_transaction()
        del tx["TransactionAmt"]
        assert client.post("/score", json=tx).status_code == 422

    def test_wrong_type_rejected(self, client: TestClient) -> None:
        tx = valid_transaction()
        tx["TransactionAmt"] = "not-a-number"
        assert client.post("/score", json=tx).status_code == 422

    def test_unknown_fields_ignored(self, client: TestClient) -> None:
        tx = valid_transaction() | {"V999": 1.0}
        assert client.post("/score", json=tx).status_code == 200

    def test_no_champion_returns_503(self, tmp_path, monkeypatch) -> None:
        # empty registry: no champion alias anywhere
        uri = f"sqlite:///{(tmp_path / 'empty.db').as_posix()}"
        app = create_app(tracking_uri=uri, model_name="never-registered", database_dsn=None)
        with TestClient(app) as tc:
            assert tc.post("/score", json=valid_transaction()).status_code == 503
            assert tc.get("/health").json()["model_loaded"] is False


class TestBatch:
    def test_batch_scores_each_transaction(self, client: TestClient, sample_transactions) -> None:
        response = client.post("/score/batch", json={"transactions": sample_transactions})
        assert response.status_code == 200
        body = response.json()
        assert len(body["results"]) == 3
        assert all(r["decision"] in (ALERT, APPROVE) for r in body["results"])

    def test_batch_above_limit_rejected(self, client: TestClient) -> None:
        txs = [valid_transaction()] * 1001
        assert client.post("/score/batch", json={"transactions": txs}).status_code == 422

    def test_batch_empty_rejected_or_accepted_consistently(self, client: TestClient) -> None:
        response = client.post("/score/batch", json={"transactions": []})
        assert response.status_code == 200
        assert response.json()["results"] == []


class TestOps:
    def test_health_and_model(self, client: TestClient) -> None:
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["model_loaded"] is True
        model = client.get("/model").json()
        assert model["version"] == 1
        assert model["threshold"] >= 0
        assert model["train_window"]["train_days"] == 20
        assert "validation" in model["metrics"]

    def test_metrics_prometheus_format(self, client: TestClient) -> None:
        client.post("/score", json=valid_transaction())
        response = client.get("/metrics")
        assert response.status_code == 200
        body = response.text
        assert "fraudops_http_requests_total" in body
        assert "fraudops_score_distribution" in body
        assert 'decision="alert"' in body or 'decision="approve"' in body

    def test_admin_reload_requires_token(self, client: TestClient) -> None:
        assert client.post("/admin/reload").status_code == 401
        assert client.post("/admin/reload", headers={"X-Admin-Token": "wrong"}).status_code == 401
        ok = client.post("/admin/reload", headers={"X-Admin-Token": "test-token"})
        assert ok.status_code == 200
        assert ok.json()["model_version"] == 1


class TestPersistence:
    def test_every_score_is_enqueued_with_version(
        self, client: TestClient, sample_transactions
    ) -> None:
        client.post("/score", json=valid_transaction())
        client.post("/score/batch", json={"transactions": sample_transactions})
        sink: CapturingSink = client.app.state.service.sink  # type: ignore[attr-defined]
        assert len(sink.captured) == 4
        assert all(p.model_version == 1 for p in sink.captured)
        assert all(p.decision in (ALERT, APPROVE) for p in sink.captured)
        assert all(0.0 <= p.fraud_probability <= 1.0 for p in sink.captured)


class TestTrainServeParity:
    def test_api_score_equals_direct_pipeline_score(self, client: TestClient, registry) -> None:
        """The single-pipeline guarantee: serving path == training library path."""
        import mlflow

        mlflow.set_tracking_uri(registry["uri"])  # models:/ URIs use the global config
        pyfunc = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
        bundle = FraudOpsModel.unwrap(pyfunc)
        tx = valid_transaction()
        body = client.post("/score", json=tx).json()
        direct = float(bundle.score_raw(pd.DataFrame([tx]))[0])
        assert body["fraud_probability"] == pytest.approx(direct, abs=1e-9)


class TestNativeTreeShap:
    def test_pred_contrib_matches_shap_tree_explainer(self, trained_bundle) -> None:
        """The serving path's LightGBM-native TreeSHAP must equal
        shap.TreeExplainer (the batch/report path) value for value."""
        import numpy as np
        import shap

        bundle, frame = trained_bundle
        X = bundle.pipeline.transform(frame.head(30))
        native = bundle.model.booster_.predict(X, pred_contrib=True)[:, :-1]
        reference = np.asarray(shap.TreeExplainer(bundle.model.booster_).shap_values(X))
        if reference.ndim == 3:
            reference = reference[..., -1]
        np.testing.assert_allclose(native, reference, rtol=1e-6, atol=1e-8)

    def test_sigmoid_of_contributions_equals_probability(self, trained_bundle) -> None:
        """sum(contributions + base) -> logistic == booster.predict."""
        import math

        bundle, frame = trained_bundle
        X = bundle.pipeline.transform(frame.head(30))
        contrib = bundle.model.booster_.predict(X, pred_contrib=True)
        for row, expected in zip(contrib, bundle.model.booster_.predict(X), strict=True):
            assert 1.0 / (1.0 + math.exp(-row.sum())) == pytest.approx(expected, abs=1e-9)

    def test_transform_handles_loader_category_dtypes(self, trained_bundle) -> None:
        """Training frames arrive from the loader with pandas category
        columns — the cached-map encoding must handle them too."""
        bundle, frame = trained_bundle
        categorical = frame[["ProductCD", "card4", "DeviceInfo"]].astype("category")
        typed = frame.assign(**{c: categorical[c] for c in categorical.columns})
        out = bundle.pipeline.transform(typed.head(10))
        reference = bundle.pipeline.transform(frame.head(10))
        pd.testing.assert_frame_equal(out, reference)
