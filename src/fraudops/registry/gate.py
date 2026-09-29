"""Champion/challenger promotion gate.

A challenger is promoted to the ``champion`` alias **only if**, on the most
recent fully-labelled window, it (a) beats the champion's expected cost per
100k transactions by at least a configurable margin, and (b) does not have a
lower PR-AUC. Every decision — promote or reject — is logged to the
``promotion_log`` table and to MLflow, so a promotion is auditable and a
rollback command exists.

``evaluate_gate`` is pure (table-testable); ``apply_gate`` performs the I/O
(alias move, logging) on top of the same registry primitives the CLI uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import psycopg

from fraudops.registry import client as rc


@dataclass(frozen=True)
class WindowMetrics:
    """One model's metrics on the labelled evaluation window."""

    version: int
    pr_auc: float
    cost_per_100k: float
    n_labelled: int
    window_desc: str
    threshold: float | None = None


@dataclass(frozen=True)
class GateConfig:
    # challenger cost must be lower than champion cost by this relative margin
    # (0.01 = at least 1% cheaper); ties inside the margin are REJECTED
    min_cost_advantage: float = 0.01
    # challenger PR-AUC must not be lower than the champion's
    allow_pr_auc_drop: float = 0.0


DEFAULT_GATE_CONFIG = GateConfig()


@dataclass(frozen=True)
class GateDecision:
    promote: bool
    reason: str


def evaluate_gate(
    champion: WindowMetrics,
    challenger: WindowMetrics,
    config: GateConfig = DEFAULT_GATE_CONFIG,
) -> GateDecision:
    """Decide promotion from both models' window metrics. Pure."""
    if champion.n_labelled != challenger.n_labelled:
        raise ValueError("metrics must come from the same labelled window")

    if math.isnan(challenger.pr_auc) or math.isnan(challenger.cost_per_100k):
        return GateDecision(False, "challenger metrics undefined on the window")

    if champion.cost_per_100k <= 0:
        return GateDecision(False, "champion cost is already zero — nothing to improve")

    cost_advantage = (champion.cost_per_100k - challenger.cost_per_100k) / champion.cost_per_100k
    pr_auc_drop = champion.pr_auc - challenger.pr_auc

    if cost_advantage < config.min_cost_advantage:
        return GateDecision(
            False,
            f"cost advantage {cost_advantage:+.1%} below the required "
            f"{config.min_cost_advantage:.0%} margin "
            f"(champion {champion.cost_per_100k:,.0f} vs challenger "
            f"{challenger.cost_per_100k:,.0f} per 100k)",
        )

    if pr_auc_drop > config.allow_pr_auc_drop:
        return GateDecision(
            False,
            f"PR-AUC drop {champion.pr_auc:.3f} -> {challenger.pr_auc:.3f} "
            f"not allowed (tolerance {config.allow_pr_auc_drop})",
        )

    return GateDecision(
        True,
        f"challenger wins: cost {champion.cost_per_100k:,.0f} -> "
        f"{challenger.cost_per_100k:,.0f} per 100k ({cost_advantage:+.1%}), "
        f"PR-AUC {champion.pr_auc:.3f} -> {challenger.pr_auc:.3f}",
    )


def apply_gate(
    client,
    database_dsn: str,
    champion: WindowMetrics,
    challenger: WindowMetrics,
    config: GateConfig = DEFAULT_GATE_CONFIG,
    connect=psycopg.connect,
) -> GateDecision:
    """Evaluate the gate, move the alias when it passes, log either way.

    ``connect`` is injectable so tests can intercept the promotion_log write.
    """
    decision = evaluate_gate(champion, challenger, config)

    if decision.promote:
        rc.set_alias(client, rc.CHAMPION, str(challenger.version))
        # the promoted model is the new champion: no challenger stands
        client.delete_registered_model_alias(rc.model_name(), rc.CHALLENGER)

    with connect(database_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
                INSERT INTO promotion_log
                    (champion_version, challenger_version, champion_pr_auc,
                     champion_cost_100k, challenger_pr_auc, challenger_cost_100k,
                     n_labelled, window_desc, decision, reason)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
            (
                champion.version,
                challenger.version,
                champion.pr_auc,
                champion.cost_per_100k,
                challenger.pr_auc,
                challenger.cost_per_100k,
                champion.n_labelled,
                champion.window_desc,
                "promoted" if decision.promote else "rejected",
                decision.reason,
            ),
        )

    print(
        f"gate: challenger v{challenger.version} vs champion v{champion.version} "
        f"on {champion.window_desc} -> {'PROMOTED' if decision.promote else 'REJECTED'}"
        f" — {decision.reason}"
    )
    return decision
