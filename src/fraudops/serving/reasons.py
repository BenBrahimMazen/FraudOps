"""Reason codes for flagged transactions (analyst-facing explanations).

Contributions come from LightGBM's native TreeSHAP implementation
(``pred_contrib=True``): the same algorithm as ``shap.TreeExplainer`` for
tree models, computed in C during the prediction pass (one call for both
the score and its reasons). Equivalence with shap.TreeExplainer is locked
by a unit test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reasons_from_contributions(
    contributions: np.ndarray, X_row: pd.DataFrame, k: int = 3
) -> list[dict]:
    """Top-k signed contributions for one transformed row.

    ``contributions`` is the per-feature TreeSHAP vector (base value excluded).
    Positive pushes toward fraud (log-odds space).
    """
    order = np.argsort(-np.abs(contributions))[:k]
    reasons = []
    for i in order:
        raw = X_row.iloc[0, i]
        reasons.append(
            {
                "feature": str(X_row.columns[i]),
                "value": _json_value(raw),
                "contribution": float(contributions[i]),
            }
        )
    return reasons


def _json_value(raw) -> float | int | str | None:
    """Plain JSON types for the reason payload (numpy -> builtins)."""
    if raw is None or raw is pd.NA or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, np.floating):
        return float(raw)
    if isinstance(raw, np.integer):
        return int(raw)
    if isinstance(raw, str):
        return raw
    return str(raw)
