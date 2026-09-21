"""Cost function and cost-optimal threshold tests, including edge cases."""

import numpy as np
import pytest

from fraudops.models.cost import cost_at_threshold, cost_per_100k, total_cost
from fraudops.models.threshold import NO_ALERT_THRESHOLD, optimal_cost_threshold

FAC = 5.0  # false alert cost


class TestCost:
    def test_hand_computed_example(self) -> None:
        # one missed fraud of 100, two false alerts
        y = np.array([1, 0, 0, 0])
        amt = np.array([100.0, 10.0, 20.0, 30.0])
        flagged = np.array([0, 1, 1, 0])
        assert total_cost(y, amt, flagged, FAC) == pytest.approx(100.0 + 2 * FAC)

    def test_perfect_decisions_cost_zero(self) -> None:
        y = np.array([1, 0])
        amt = np.array([500.0, 10.0])
        flagged = np.array([1, 0])
        assert total_cost(y, amt, flagged, FAC) == 0.0

    def test_no_fraud_window_costs_only_false_alerts(self) -> None:
        y = np.zeros(4, dtype=int)
        amt = np.array([10.0, 20.0, 30.0, 40.0])
        flagged = np.array([1, 1, 0, 0])
        assert total_cost(y, amt, flagged, FAC) == pytest.approx(2 * FAC)

    def test_all_fraud_window_costs_only_misses(self) -> None:
        y = np.ones(3, dtype=int)
        amt = np.array([10.0, 20.0, 30.0])
        flagged = np.array([0, 0, 1])
        assert total_cost(y, amt, flagged, FAC) == pytest.approx(30.0)

    def test_cost_per_100k_normalisation(self) -> None:
        # 1 missed fraud of 100 among 1000 rows -> 10_000 per 100k
        y = np.zeros(1000, dtype=int)
        y[0] = 1
        amt = np.zeros(1000)
        amt[0] = 100.0
        assert cost_per_100k(y, amt, np.zeros(1000, dtype=int), FAC) == pytest.approx(10_000.0)

    def test_cost_at_threshold_flags_at_or_above(self) -> None:
        y = np.array([1, 0])
        amt = np.array([100.0, 1.0])
        scores = np.array([0.7, 0.3])
        # flag only the fraud: threshold anywhere in (0.3, 0.7]
        assert cost_at_threshold(y, amt, scores, 0.7, FAC) == 0.0
        assert cost_at_threshold(y, amt, scores, 0.71, FAC) == pytest.approx(100.0)


class TestOptimalThreshold:
    def test_obvious_case_flags_only_large_frauds(self) -> None:
        # two frauds: 1000 (high score) and 1 (low score); many legit rows.
        # Alerting the 1-amount fraud costs 5 > 1 -> optimal misses it.
        y = np.array([1, 1, 0, 0, 0, 0])
        amt = np.array([1000.0, 1.0, 10.0, 10.0, 10.0, 10.0])
        scores = np.array([0.9, 0.2, 0.1, 0.1, 0.1, 0.1])
        t = optimal_cost_threshold(y, amt, scores, FAC)
        assert 0.2 < t <= 0.9
        assert cost_at_threshold(y, amt, scores, t, FAC) == pytest.approx(1.0)

    def test_no_fraud_alerts_nothing(self) -> None:
        y = np.zeros(50, dtype=int)
        amt = np.full(50, 50.0)
        scores = np.linspace(0.01, 0.99, 50)
        t = optimal_cost_threshold(y, amt, scores, FAC)
        assert t == NO_ALERT_THRESHOLD
        assert cost_at_threshold(y, amt, scores, t, FAC) == 0.0

    def test_all_fraud_large_amounts_alerts_everything(self) -> None:
        y = np.ones(10, dtype=int)
        amt = np.full(10, 500.0)  # every fraud worth more than FAC
        scores = np.linspace(0.05, 0.95, 10)
        t = optimal_cost_threshold(y, amt, scores, FAC)
        assert t == pytest.approx(scores.min())

    def test_ties_prefer_higher_threshold(self) -> None:
        # two thresholds with identical cost -> fewer alerts wins
        y = np.array([1, 0, 0, 0])
        amt = np.array([100.0, 1.0, 1.0, 1.0])
        scores = np.array([0.9, 0.5, 0.6, 0.7])
        # catching the fraud costs 1*FAC=5; missing it costs 100.
        # any threshold in (0.5, 0.9] costs 5 -> pick the highest, 0.9.
        t = optimal_cost_threshold(y, amt, scores, FAC)
        assert t == pytest.approx(0.9)

    def test_matches_bruteforce_on_random_case(self) -> None:
        rng = np.random.default_rng(7)
        y = rng.integers(0, 2, size=500)
        amt = rng.uniform(1, 800, size=500)
        scores = rng.uniform(0, 1, size=500)
        best = optimal_cost_threshold(y, amt, scores, FAC)
        grid = np.unique(scores)
        costs = [cost_at_threshold(y, amt, scores, g, FAC) for g in grid]
        costs.append(cost_at_threshold(y, amt, scores, NO_ALERT_THRESHOLD, FAC))
        assert cost_at_threshold(y, amt, scores, best, FAC) == pytest.approx(min(costs))

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            optimal_cost_threshold(np.zeros(3), np.zeros(3), np.zeros(4), FAC)
