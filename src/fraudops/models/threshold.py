"""Cost-optimal decision threshold.

The threshold minimises expected total cost on validation data (not F1). The
search is exact over the observed score values, vectorised, and implemented as
a pure function so it can be table-tested and reused by monitoring windows.
"""

from __future__ import annotations

import numpy as np

# Sentinel returned when the cost-optimal action is "alert on nothing"
# (i.e. the optimum lies above every observed score). Just above 1.0 so that
# `scores >= threshold` is false even for a score of exactly 1.0.
NO_ALERT_THRESHOLD = float(np.nextafter(1.0, 2.0))


def optimal_cost_threshold(
    y_true: np.ndarray,
    amounts: np.ndarray,
    scores: np.ndarray,
    false_alert_cost: float,
) -> float:
    """Return the threshold minimising expected cost; ties -> higher threshold.

    Edge cases (tested):
    - no positives and any negatives: alerting can only add false-alert cost,
      so the optimum is to alert on nothing -> NO_ALERT_THRESHOLD.
    - all positives: the first fraud amount above the false-alert cost makes
      alerting everything optimal -> lowest observed score.
    """
    y_true = np.asarray(y_true).astype(bool)
    amounts = np.asarray(amounts, dtype="float64")
    scores = np.asarray(scores, dtype="float64")
    if len(scores) != len(y_true) or len(scores) != len(amounts):
        raise ValueError("y_true, amounts and scores must have equal length")
    if len(scores) == 0:
        return NO_ALERT_THRESHOLD

    # Candidates: every distinct score (flagging "score >= s" changes exactly
    # at observed values). Sorting ascending lets prefix sums track the cost.
    order = np.argsort(scores, kind="stable")
    s_sorted = scores[order]
    y_sorted = y_true[order]
    amt_sorted = amounts[order]

    # caught fraud amount if everything up to each prefix is alerted
    fraud_amt_cum = np.cumsum(np.where(y_sorted, amt_sorted, 0.0))
    total_fraud_amt = float(fraud_amt_cum[-1])

    # For each distinct score value u (as threshold): rows with score >= u.
    # With ascending sort these are rows [k, n) where k = first index of u.
    _, first_idx = np.unique(s_sorted, return_index=True)
    caught = fraud_amt_cum[-1] - np.where(first_idx > 0, fraud_amt_cum[first_idx - 1], 0.0)
    alerts = len(s_sorted) - first_idx
    costs = (total_fraud_amt - caught) + false_alert_cost * alerts

    # Candidate "alert on nothing" (threshold above the max score).
    costs = np.append(costs, total_fraud_amt)
    thresholds = np.append(s_sorted[first_idx], NO_ALERT_THRESHOLD)

    # Min cost; among equal costs prefer the HIGHER threshold (fewer alerts).
    best = int(np.argmax(np.where(costs == costs.min(), thresholds, -np.inf)))
    return float(thresholds[best])
