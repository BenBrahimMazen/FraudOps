"""Retrain trigger: a pure, table-testable decision function.

Rules (configurable via configs/drift.yaml, defaults from the project spec):
retrain when ANY of
- score-distribution PSI is at alert level,
- at least N of the top-K features are at alert level,
- performance degraded beyond tolerance on the labelled window
  (PR-AUC drop beyond tolerance, or expected cost ratio vs the reference
  window exceeding its tolerance).

The function receives evidence, returns a decision with the reasons that
fired. Phase 4's Airflow DAG calls exactly this function.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fraudops.monitoring.drift import DriftResult, Level


@dataclass(frozen=True)
class PerformanceSnapshot:
    """Metrics on the most recent fully-labelled window."""

    pr_auc: float | None
    cost_per_100k: float
    n_labelled: int


@dataclass(frozen=True)
class TriggerConfig:
    score_psi_alert: float = 0.2
    feature_psi_alert: float = 0.2
    min_features_in_alert: int = 3
    pr_auc_tolerance: float = 0.05  # absolute drop vs reference
    cost_ratio_tolerance: float = 1.15  # window cost / reference cost
    min_labelled: int = 500  # too few labels -> performance rule stays silent


@dataclass(frozen=True)
class TriggerDecision:
    retrain: bool
    reasons: list[str] = field(default_factory=list)


DEFAULT_TRIGGER_CONFIG = TriggerConfig()


def should_retrain(
    score_drift: DriftResult | None,
    feature_drifts: list[DriftResult],
    performance: PerformanceSnapshot | None,
    reference_pr_auc: float | None = None,
    reference_cost_per_100k: float | None = None,
    config: TriggerConfig = DEFAULT_TRIGGER_CONFIG,
) -> TriggerDecision:
    """Evidence in, decision out. Pure: no I/O, deterministic."""
    reasons: list[str] = []

    if score_drift is not None and score_drift.psi > config.score_psi_alert:
        reasons.append(f"score PSI {score_drift.psi:.3f} > {config.score_psi_alert}")

    alerting = [d for d in feature_drifts if d.psi > config.feature_psi_alert]
    if len(alerting) >= config.min_features_in_alert:
        names = ", ".join(f"{d.name} ({d.psi:.2f})" for d in alerting[:5])
        reasons.append(f"{len(alerting)} features above PSI {config.feature_psi_alert}: {names}")

    if (
        performance is not None
        and performance.n_labelled >= config.min_labelled
        and reference_pr_auc is not None
        and performance.pr_auc is not None
        and (reference_pr_auc - performance.pr_auc) > config.pr_auc_tolerance
    ):
        reasons.append(
            f"PR-AUC dropped {reference_pr_auc:.3f} -> {performance.pr_auc:.3f} "
            f"(beyond {config.pr_auc_tolerance})"
        )

    if (
        performance is not None
        and performance.n_labelled >= config.min_labelled
        and reference_cost_per_100k is not None
        and reference_cost_per_100k > 0
        and performance.cost_per_100k / reference_cost_per_100k > config.cost_ratio_tolerance
    ):
        reasons.append(
            f"cost per 100k {performance.cost_per_100k:,.0f} vs reference "
            f"{reference_cost_per_100k:,.0f} (ratio beyond {config.cost_ratio_tolerance})"
        )

    return TriggerDecision(retrain=bool(reasons), reasons=reasons)


def summarize(results: list[DriftResult]) -> dict:
    """Counts by level, for logging and dashboards."""
    counts = {level.value: 0 for level in Level}
    for result in results:
        counts[result.level.value] += 1
    return counts
