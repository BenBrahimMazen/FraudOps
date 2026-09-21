"""Evaluation: primary business metrics plus reference baselines.

Primary metrics (per the project framing):
- PR-AUC (average precision) — the imbalance-aware ranking metric
- recall at a fixed false-positive rate (default 1%)
- expected cost per 100k transactions

Reference-only: ROC-AUC and F1. Baselines compared against the model:
(a) no model (approve everything), (b) fixed 0.5 threshold,
(c) F1-optimal threshold.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from fraudops.models.cost import cost_per_100k, total_cost


def recall_at_fpr(
    y_true: np.ndarray, scores: np.ndarray, fpr_target: float = 0.01
) -> tuple[float, float]:
    """Recall achieved at (at most) the given false-positive rate.

    Returns (recall, threshold): the operating point on the ROC curve with the
    largest threshold whose FPR stays <= fpr_target.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    mask = fpr <= fpr_target
    if not mask.any():
        return 0.0, float(np.inf)
    i = int(np.nonzero(mask)[0][-1])  # thresholds decrease as fpr increases
    return float(tpr[i]), float(thresholds[i])


def f1_optimal_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising F1 (the reference decision rule, not the primary)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.nanargmax(f1))])


def compute_metrics(
    y_true: np.ndarray,
    amounts: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    false_alert_cost: float,
    fpr_target: float = 0.01,
) -> dict[str, float]:
    """Full metric dictionary for one window at one decision threshold."""
    flagged = np.asarray(scores) >= threshold
    recall01, _ = recall_at_fpr(y_true, scores, fpr_target)
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "recall_at_fpr": recall01,
        "f1": float(f1_score(y_true, flagged, zero_division=0)),
        "alert_rate": float(flagged.mean()),
        "cost_per_100k": cost_per_100k(y_true, amounts, flagged, false_alert_cost),
    }


def baseline_costs(
    y_true: np.ndarray,
    amounts: np.ndarray,
    scores: np.ndarray,
    false_alert_cost: float,
) -> dict[str, float]:
    """Cost per 100k of the three reference strategies.

    - no_model: approve everything (all fraud missed, zero alerts)
    - fixed_0.5: flag score >= 0.5
    - f1_optimal: flag at the F1-maximising threshold
    """
    y_true = np.asarray(y_true).astype(bool)
    amounts = np.asarray(amounts, dtype="float64")
    scores = np.asarray(scores)
    n = max(len(y_true), 1)

    no_model = total_cost(y_true, amounts, np.zeros(len(y_true), dtype=bool), false_alert_cost)
    fixed = total_cost(y_true, amounts, scores >= 0.5, false_alert_cost)
    f1_t = f1_optimal_threshold(y_true, scores)
    f1_cost = total_cost(y_true, amounts, scores >= f1_t, false_alert_cost)

    def per_100k(cost: float) -> float:
        return float(cost) / n * 100_000

    return {
        "no_model": per_100k(no_model),
        "fixed_0.5": per_100k(fixed),
        "f1_optimal": per_100k(f1_cost),
    }
