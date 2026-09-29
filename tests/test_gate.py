"""Champion/challenger gate: promote only on a proven cost win, never on PR-AUC loss.

Table-driven tests over the pure decision function, plus alias/DB side effects
of ``apply_gate`` against the throwaway registry.
"""

from __future__ import annotations

import mlflow
import pytest

from fraudops.registry import client as rc
from fraudops.registry.gate import (
    DEFAULT_GATE_CONFIG,
    GateConfig,
    WindowMetrics,
    apply_gate,
    evaluate_gate,
)

# champion baseline for the table: 10_000 per 100k, PR-AUC 0.80, 5_000 labelled
CHAMPION = WindowMetrics(
    version=1, pr_auc=0.80, cost_per_100k=10_000.0, n_labelled=5_000, window_desc="days 170-177"
)


def challenger(cost: float, pr_auc: float = 0.80) -> WindowMetrics:
    return WindowMetrics(
        version=2, pr_auc=pr_auc, cost_per_100k=cost, n_labelled=5_000, window_desc="days 170-177"
    )


@pytest.mark.parametrize(
    ("challenger_metrics", "expected_promote", "label"),
    [
        # clear win: 20% cheaper, same PR-AUC
        (challenger(8_000.0), True, "better_challenger_promotes"),
        # same cost, same PR-AUC: tie inside the margin is a reject
        (challenger(10_000.0), False, "tie_rejected"),
        # just inside the 1% margin: reject (0.5% advantage)
        (challenger(9_950.0), False, "within_margin_rejected"),
        # exactly at the margin: promote (>= boundary)
        (challenger(9_900.0), True, "exactly_at_margin_promotes"),
        # more expensive: reject
        (challenger(11_000.0), False, "worse_challenger_rejected"),
        # cheaper but worse PR-AUC: the cost win does not buy a ranking loss
        (challenger(8_000.0, pr_auc=0.79), False, "pr_auc_loss_blocks_promotion"),
        # cheaper AND better PR-AUC
        (challenger(8_000.0, pr_auc=0.82), True, "wins_on_both_promotes"),
        # undefined metrics (e.g. no positives in the window)
        (challenger(float("nan"), pr_auc=float("nan")), False, "nan_metrics_rejected"),
    ],
)
def test_gate_table(challenger_metrics, expected_promote, label) -> None:
    decision = evaluate_gate(CHAMPION, challenger_metrics)
    assert decision.promote is expected_promote, label
    assert decision.reason  # every decision carries an operator-readable reason


def test_custom_margin_changes_the_boundary() -> None:
    # 2% advantage with a 1% required margin: promote...
    assert evaluate_gate(CHAMPION, challenger(9_800.0)).promote
    # ...but not when 5% is required
    stricter = GateConfig(min_cost_advantage=0.05)
    assert not evaluate_gate(CHAMPION, challenger(9_800.0), stricter).promote


def test_pr_auc_tolerance_can_allow_a_small_drop() -> None:
    tolerant = GateConfig(allow_pr_auc_drop=0.02)
    assert evaluate_gate(CHAMPION, challenger(8_000.0, pr_auc=0.79), tolerant).promote


def test_zero_cost_champion_cannot_be_beaten() -> None:
    perfect = WindowMetrics(
        version=1, pr_auc=0.99, cost_per_100k=0.0, n_labelled=5_000, window_desc="days 170-177"
    )
    assert not evaluate_gate(perfect, challenger(-1.0)).promote


def test_mismatched_windows_raise() -> None:
    other_window = WindowMetrics(
        version=2, pr_auc=0.9, cost_per_100k=1.0, n_labelled=4_999, window_desc="days 168-175"
    )
    with pytest.raises(ValueError, match="same labelled window"):
        evaluate_gate(CHAMPION, other_window)


# ---------------------------------------------------------------- apply_gate


class FakeCursor:
    def __init__(self, statements: list) -> None:
        self._statements = statements

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None) -> None:
        self._statements.append((sql, params))


class FakeConn:
    """Records INSERTs; stands in for psycopg against the promotion_log."""

    def __init__(self) -> None:
        self.statements: list = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return FakeCursor(self.statements)


def _apply(registry, champion: WindowMetrics, challenger_m: WindowMetrics, dsn="unused"):
    fake = FakeConn()
    decision = apply_gate(registry["client"], dsn, champion, challenger_m, connect=lambda dsn: fake)
    return decision, fake


def _register_new_version(registry) -> int:
    """Register another real version from a fresh run (v1 exists already)."""
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
    mv = registry["client"].create_model_version(
        rc.model_name(), source=f"runs:/{run_id}/model", run_id=run_id
    )
    return int(mv.version)


def test_apply_gate_promotes_and_moves_alias(registry) -> None:
    new_version = _register_new_version(registry)
    cli = registry["client"]
    rc.set_alias(cli, rc.CHALLENGER, str(new_version))

    champ = WindowMetrics(1, 0.80, 10_000.0, 5_000, "days 170-177")
    chall = WindowMetrics(new_version, 0.81, 8_000.0, 5_000, "days 170-177")
    decision, fake = _apply(registry, champ, chall)

    assert decision.promote
    assert rc.get_version_by_alias(cli, rc.CHAMPION) == str(new_version)
    assert rc.get_version_by_alias(cli, rc.CHALLENGER) is None  # cleared on promote
    sql, params = fake.statements[0]
    assert "INSERT INTO promotion_log" in sql
    assert params[-2] == "promoted"
    assert params[0] == 1 and params[1] == new_version


def test_apply_gate_rejects_and_leaves_aliases(registry) -> None:
    new_version = _register_new_version(registry)
    cli = registry["client"]
    rc.set_alias(cli, rc.CHALLENGER, str(new_version))
    champion_before = rc.get_version_by_alias(cli, rc.CHAMPION)

    champ = WindowMetrics(registry["version"], 0.80, 10_000.0, 5_000, "days 170-177")
    chall = WindowMetrics(new_version, 0.80, 10_500.0, 5_000, "days 170-177")
    decision, fake = _apply(registry, champ, chall)

    assert not decision.promote
    assert rc.get_version_by_alias(cli, rc.CHAMPION) == champion_before
    assert rc.get_version_by_alias(cli, rc.CHALLENGER) == str(new_version)
    _, params = fake.statements[0]
    assert params[-2] == "rejected"


def test_default_margin_is_one_percent() -> None:
    assert DEFAULT_GATE_CONFIG.min_cost_advantage == 0.01
    assert DEFAULT_GATE_CONFIG.allow_pr_auc_drop == 0.0
