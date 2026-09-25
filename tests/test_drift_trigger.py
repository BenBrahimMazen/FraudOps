"""PSI/KS known-answer tests and table-driven retrain-trigger tests."""

import numpy as np
import pytest

from fraudops.monitoring.drift import (
    Level,
    evaluate_drift,
    fit_numeric_bins,
    ks_pvalue,
    level_for,
    psi_categorical,
    psi_numeric,
)
from fraudops.monitoring.trigger import (
    PerformanceSnapshot,
    TriggerConfig,
    should_retrain,
)

RNG = np.random.default_rng(7)


class TestPsiNumeric:
    def test_same_distribution_psi_near_zero(self) -> None:
        a = RNG.normal(0, 1, 20_000)
        b = RNG.normal(0, 1, 20_000)
        assert psi_numeric(a, b) < 0.02

    def test_shifted_mean_psi_large(self) -> None:
        a = RNG.normal(0, 1, 20_000)
        b = RNG.normal(2.0, 1, 20_000)
        assert psi_numeric(a, b) > 1.0

    def test_scale_shift_detected(self) -> None:
        a = RNG.lognormal(0, 1, 20_000)
        b = RNG.lognormal(0, 1.6, 20_000)
        assert psi_numeric(a, b) > 0.2

    def test_moderate_shift_lands_in_warning_band(self) -> None:
        a = RNG.normal(0, 1, 50_000)
        b = RNG.normal(0.35, 1, 50_000)
        value = psi_numeric(a, b)
        assert 0.1 < value <= 0.25  # warning band territory

    def test_bins_fitted_on_reference_are_stable(self) -> None:
        reference = RNG.normal(0, 1, 5_000)
        edges = fit_numeric_bins(reference, bins=10)
        assert edges[0] == -np.inf and edges[-1] == np.inf
        # reused edges give the same PSI for identical samples
        same = RNG.normal(0, 1, 5_000)
        assert psi_numeric(reference, same, edges=edges) < 0.02

    def test_empty_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            psi_numeric(np.array([1.0, 2.0]), np.array([]))


class TestPsiCategorical:
    def test_same_mix_near_zero(self) -> None:
        a = RNG.choice(["x", "y", "z"], size=10_000, p=[0.5, 0.3, 0.2])
        b = RNG.choice(["x", "y", "z"], size=10_000, p=[0.5, 0.3, 0.2])
        assert psi_categorical(a, b) < 0.02

    def test_remixed_categories_detected(self) -> None:
        a = RNG.choice(["x", "y"], size=10_000, p=[0.5, 0.5])
        b = RNG.choice(["x", "y"], size=10_000, p=[0.9, 0.1])
        assert psi_categorical(a, b) > 0.5

    def test_unseen_category_counts_as_drift(self) -> None:
        a = RNG.choice(["x", "y"], size=5_000, p=[0.5, 0.5])
        b = np.array(["x"] * 2500 + ["NEW"] * 2500, dtype=object)
        assert psi_categorical(a, b) > 0.2


class TestKs:
    def test_identical_samples_high_pvalue(self) -> None:
        a = RNG.normal(0, 1, 2_000)
        assert ks_pvalue(a, a.copy()) > 0.5

    def test_shifted_samples_low_pvalue(self) -> None:
        a = RNG.normal(0, 1, 2_000)
        b = RNG.normal(1.5, 1, 2_000)
        assert ks_pvalue(a, b) < 1e-6

    def test_too_few_samples_returns_none(self) -> None:
        assert ks_pvalue(np.array([1.0]), np.array([1.0])) is None


class TestLevels:
    def test_threshold_boundaries(self) -> None:
        assert level_for(0.05) is Level.OK
        assert level_for(0.15) is Level.WARNING
        assert level_for(0.25) is Level.ALERT

    def test_evaluate_drift_dispatches_numeric_vs_categorical(self) -> None:
        numeric = evaluate_drift("amt", RNG.normal(0, 1, 5_000), RNG.normal(3, 1, 5_000))
        assert numeric.level is Level.ALERT and numeric.ks_pvalue is not None
        cats = evaluate_drift(
            "cd",
            np.array(["a"] * 500 + ["b"] * 500, dtype=object),
            np.array(["a"] * 950 + ["b"] * 50, dtype=object),
        )
        assert cats.ks_pvalue is None


def _drift(name: str, psi_value: float) -> "object":
    from fraudops.monitoring.drift import DriftResult

    return DriftResult(name=name, psi=psi_value, ks_pvalue=None, level=level_for(psi_value))


class TestShouldRetrain:
    CFG = TriggerConfig()

    # name, score, features, perf, ref_pr_auc, ref_cost, expect_retrain
    CASES = [
        ("quiet-stream", None, [], None, None, None, False),
        (
            "score-alert",
            _drift("score", 0.31),
            [],
            None,
            None,
            None,
            True,
        ),
        (
            "score-warning-only",
            _drift("score", 0.15),
            [],
            None,
            None,
            None,
            False,
        ),
        (
            "three-features-alert",
            None,
            [_drift("f1", 0.3), _drift("f2", 0.25), _drift("f3", 0.4)],
            None,
            None,
            None,
            True,
        ),
        (
            "two-features-alert-not-enough",
            None,
            [_drift("f1", 0.3), _drift("f2", 0.25)],
            None,
            None,
            None,
            False,
        ),
        (
            "pr-auc-drop",
            None,
            [],
            PerformanceSnapshot(pr_auc=0.35, cost_per_100k=200_000, n_labelled=2_000),
            0.55,
            210_000,
            True,
        ),
        (
            "pr-auc-small-drop-ignored",
            None,
            [],
            PerformanceSnapshot(pr_auc=0.53, cost_per_100k=200_000, n_labelled=2_000),
            0.55,
            210_000,
            False,
        ),
        (
            "cost-spike",
            None,
            [],
            PerformanceSnapshot(pr_auc=0.54, cost_per_100k=400_000, n_labelled=2_000),
            0.55,
            210_000,
            True,
        ),
        (
            "few-labels-silence-performance-rule",
            None,
            [],
            PerformanceSnapshot(pr_auc=0.10, cost_per_100k=900_000, n_labelled=100),
            0.55,
            210_000,
            False,
        ),
        (
            "everything-at-once",
            _drift("score", 0.5),
            [_drift("f1", 0.4), _drift("f2", 0.4), _drift("f3", 0.4)],
            PerformanceSnapshot(pr_auc=0.20, cost_per_100k=500_000, n_labelled=5_000),
            0.55,
            210_000,
            True,
        ),
    ]

    @pytest.mark.parametrize(
        "name,score,features,perf,ref_auc,ref_cost,expected",
        CASES,
        ids=[c[0] for c in CASES],
    )
    def test_table(self, name, score, features, perf, ref_auc, ref_cost, expected) -> None:
        decision = should_retrain(score, features, perf, ref_auc, ref_cost, config=self.CFG)
        assert decision.retrain is expected, decision.reasons
        if expected:
            assert decision.reasons  # every retrain carries its reasons
