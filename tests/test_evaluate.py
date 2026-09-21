"""Evaluation metric tests against constructions with known answers."""

import numpy as np
import pytest

from fraudops.models.evaluate import (
    baseline_costs,
    compute_metrics,
    f1_optimal_threshold,
    recall_at_fpr,
)


def test_perfect_ranking_metrics() -> None:
    y = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    m = compute_metrics(y, np.full(6, 100.0), scores, threshold=0.5, false_alert_cost=5.0)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["recall_at_fpr"] == pytest.approx(1.0)
    assert m["cost_per_100k"] == pytest.approx(0.0)


def test_recall_at_fpr_known_point() -> None:
    # 100 negatives, 10 positives; only half the positives rank above
    # all negatives -> at 1% FPR (1 negative) recall tops out near 0.5.
    y = np.array([0] * 100 + [1] * 10)
    scores = np.concatenate(
        [np.linspace(0.0, 0.89, 100), np.linspace(0.91, 0.99, 5), np.linspace(0.05, 0.20, 5)]
    )
    recall, _ = recall_at_fpr(y, scores, fpr_target=0.01)
    assert recall == pytest.approx(0.5, abs=0.01)


def test_f1_optimal_threshold_found() -> None:
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    t = f1_optimal_threshold(y, scores)
    flagged = scores >= t
    assert flagged.tolist() == [False, False, False, True, True]


def test_baseline_costs_ordering_on_separable_case() -> None:
    # model separates perfectly; no-model must be the most expensive strategy
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=1000)
    amt = rng.uniform(1, 500, size=1000)
    scores = np.where(y == 1, 0.95, 0.05)
    costs = baseline_costs(y, amt, scores, false_alert_cost=5.0)
    assert costs["no_model"] > costs["fixed_0.5"] == costs["f1_optimal"] == 0.0


def test_baseline_no_model_equals_fraud_amount_per_100k() -> None:
    y = np.array([0] * 999 + [1])
    amt = np.full(1000, 10.0)
    costs = baseline_costs(y, amt, np.full(1000, 0.5), false_alert_cost=5.0)
    assert costs["no_model"] == pytest.approx(10.0 / 1000 * 100_000)
