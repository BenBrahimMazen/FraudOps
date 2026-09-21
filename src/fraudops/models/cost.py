"""Business cost model.

- A missed fraud (fraud approved) costs the transaction amount.
- A false alert (legitimate flagged) costs a fixed review effort
  (configs/costs.yaml, default 5.0).

All functions are pure and unit-tested; they are the single source of truth
for cost everywhere (training reports, monitoring, the promotion gate).
"""

from __future__ import annotations

import numpy as np


def total_cost(
    y_true: np.ndarray,
    amounts: np.ndarray,
    flagged: np.ndarray,
    false_alert_cost: float,
) -> float:
    """Total cost of binary decisions ``flagged`` (1 = alert) on one window."""
    y_true = np.asarray(y_true).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    amounts = np.asarray(amounts, dtype="float64")
    missed = amounts[y_true & ~flagged].sum()
    false_alerts = float((~y_true & flagged).sum())
    return float(missed + false_alert_cost * false_alerts)


def cost_per_100k(
    y_true: np.ndarray,
    amounts: np.ndarray,
    flagged: np.ndarray,
    false_alert_cost: float,
) -> float:
    """Expected cost normalised per 100,000 transactions."""
    n = max(len(y_true), 1)
    return total_cost(y_true, amounts, flagged, false_alert_cost) / n * 100_000


def cost_at_threshold(
    y_true: np.ndarray,
    amounts: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    false_alert_cost: float,
) -> float:
    """Total cost when flagging every transaction with score >= threshold."""
    scores = np.asarray(scores)
    return total_cost(y_true, amounts, scores >= threshold, false_alert_cost)
