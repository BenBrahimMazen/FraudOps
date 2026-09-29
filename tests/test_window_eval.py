"""Closed-loop window evaluation: injection replay, window metrics, splits."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from fraudops.models.cost import cost_per_100k
from fraudops.models.retrain import split_labelled_stream
from fraudops.models.window_eval import apply_injections_frame, evaluate_on_window
from fraudops.streaming.replay import Injector
from tests.conftest import make_training_frame

DAY = 86_400
INJECTIONS = [
    {"kind": "amount_factor", "params": {"factor": 3.0}, "from_day": 165},
    {"kind": "nullify", "params": {"columns": ["card4"]}, "from_day": 170},
]


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 60
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": np.arange(n) * DAY // 2,  # day 0..29, two rows per day
            "TransactionAmt": rng.uniform(10, 500, n).astype("float32"),
            "card4": pd.Categorical(
                rng.choice(["visa", "mastercard"], n), categories=["mastercard", "visa"]
            ),
            "C1": rng.uniform(0, 9, n).astype("float32"),
            "is_fraud": (np.arange(n) % 7 == 0).astype("int8"),
            "amount": rng.uniform(10, 500, n).astype("float64"),
        }
    )


class TestApplyInjections:
    def test_amount_factor_only_after_from_day(self) -> None:
        out = apply_injections_frame(_frame(), [INJECTIONS[0]])
        raw = _frame()
        before = out["TransactionDT"] // DAY < 165
        after = ~before
        # untouched before the injection day, multiplied from it on
        np.testing.assert_allclose(
            out.loc[before, "TransactionAmt"], raw.loc[before, "TransactionAmt"]
        )
        np.testing.assert_allclose(
            out.loc[after, "TransactionAmt"], raw.loc[after, "TransactionAmt"] * 3.0, rtol=1e-5
        )

    def test_nullify_nulls_only_chosen_columns_and_days(self) -> None:
        out = apply_injections_frame(_frame(), [INJECTIONS[1]])
        raw = _frame()
        after = out["TransactionDT"] // DAY >= 170
        assert out.loc[after, "card4"].isna().all()
        assert out.loc[~after, "card4"].notna().all()
        np.testing.assert_allclose(out["C1"], raw["C1"])  # other columns untouched

    def test_categorical_column_survives_widening(self) -> None:
        out = apply_injections_frame(_frame(), [INJECTIONS[1]])
        # widened to object for NaN assignment — non-null values keep their text
        kept = out.loc[out["card4"].notna(), "card4"]
        assert set(kept.unique()) <= {"visa", "mastercard"}

    def test_frame_matches_the_replay_producer_event_for_event(self) -> None:
        """The eval window must look EXACTLY like the stream: same injector
        rules, row by row, frame path vs the producer's dict path."""
        import math

        def _clean(v):
            if isinstance(v, float) and math.isnan(v):
                return None
            return v

        raw = _frame()
        out = apply_injections_frame(raw, INJECTIONS)
        injector = Injector(dsn=None)
        injector._injections = INJECTIONS
        injector._next_poll = time.monotonic() + 60  # skip the Postgres refresh
        for i in range(len(raw)):
            event = {k: _clean(raw.iat[i, c]) for c, k in enumerate(raw.columns)}
            day = int(event["TransactionDT"] // DAY)
            shifted = injector.apply(event, day)
            if day >= 170:
                assert shifted["card4"] is None
            if day >= 165:
                assert shifted["TransactionAmt"] == pytest.approx(
                    raw["TransactionAmt"].iat[i] * 3.0, rel=1e-5
                )
            assert shifted["TransactionAmt"] == pytest.approx(
                float(out["TransactionAmt"].iat[i]), rel=1e-5
            )


class TestEvaluateOnWindow:
    def test_extreme_thresholds_bound_the_cost(self, trained_bundle) -> None:
        """threshold 1.1 approves everything (cost = missed fraud), 0.0 flags
        everything (cost = false alerts) — proves decisions use the shipped
        threshold and cost uses the labelled amounts."""
        from fraudops.registry.wrapper import FraudOpsModel

        bundle, frame = trained_bundle
        frame = frame.copy()
        frame["amount"] = frame["TransactionAmt"].astype("float64")
        frame = frame.rename(columns={"isFraud": "is_fraud"})
        y = frame["is_fraud"].to_numpy()
        amounts = frame["amount"].to_numpy()

        approve_all = FraudOpsModel(bundle.model, bundle.pipeline, 1.1, bundle.metadata)
        flag_all = FraudOpsModel(bundle.model, bundle.pipeline, 0.0, bundle.metadata)

        m_approve = evaluate_on_window(approve_all, 1, frame, 5.0, "test window")
        m_flag = evaluate_on_window(flag_all, 1, frame, 5.0, "test window")

        expected_approve = amounts[y == 1].sum() / len(y) * 100_000
        expected_flag = 5.0 * float((y == 0).sum()) / len(y) * 100_000
        assert m_approve.cost_per_100k == pytest.approx(expected_approve, rel=1e-6)
        assert m_flag.cost_per_100k == pytest.approx(expected_flag, rel=1e-6)

    def test_pr_auc_matches_direct_computation(self, trained_bundle) -> None:
        from fraudops.models.window_eval import score_frame

        bundle, frame = trained_bundle
        frame = frame.copy()
        frame["amount"] = frame["TransactionAmt"].astype("float64")
        frame = frame.rename(columns={"isFraud": "is_fraud"})
        scores = score_frame(bundle, frame)
        metrics = evaluate_on_window(bundle, 1, frame, 5.0, "test window")
        assert metrics.pr_auc == pytest.approx(average_precision_score(frame["is_fraud"], scores))
        flagged = scores >= bundle.threshold
        assert metrics.cost_per_100k == pytest.approx(
            cost_per_100k(frame["is_fraud"], frame["amount"], flagged, 5.0)
        )
        assert metrics.n_labelled == len(frame)


class TestSplitLabelledStream:
    def test_slices_are_chronological_and_partition_the_rows(self) -> None:
        rng = np.random.default_rng(3)
        # 21 labelled days: one row per day, days 155..175 inclusive
        days = np.arange(155, 176)
        labelled = pd.DataFrame(
            {
                "TransactionID": days * 100,
                "TransactionDT": days * DAY,
                "TransactionAmt": rng.uniform(1, 99, len(days)),
                "is_fraud": (days % 5 == 0).astype("int8"),
            }
        )
        cutoff_dt = 175 * DAY
        slices = split_labelled_stream(labelled, cutoff_dt, window_sim_days=7, val_days=7)
        assert slices["gate_eval"]["TransactionDT"].min() > 168 * DAY
        assert slices["gate_eval"]["TransactionDT"].max() == cutoff_dt
        assert len(slices["gate_eval"]) == 7  # days 169..175
        assert len(slices["retrain_val"]) == 7  # days 162..168
        assert len(slices["extra_train"]) == 7  # days 155..161
        total = sum(len(p) for p in slices.values())
        assert total == len(labelled)  # no row lost or double-counted

    def test_too_short_labelled_stream_raises(self) -> None:
        labelled = pd.DataFrame(
            {
                "TransactionID": [1, 2, 3],
                "TransactionDT": [DAY * 170, DAY * 172, DAY * 174],
                "TransactionAmt": [10.0, 20.0, 30.0],
                "is_fraud": [0, 1, 0],
            }
        )
        with pytest.raises(RuntimeError, match="too short"):
            split_labelled_stream(labelled, 175 * DAY, window_sim_days=7, val_days=7)


class TestStoredDecisions:
    def test_metrics_from_served_scores_and_decisions(self) -> None:
        """The champion is judged on stored scores/decisions: PR-AUC from the
        scores, cost from the DECISIONS as served (not re-thresholded)."""
        from fraudops.models.window_eval import evaluate_stored_decisions

        y = np.array([0, 0, 1, 0, 1])
        amounts = np.array([10.0, 20.0, 100.0, 40.0, 500.0])
        scores = np.array([0.1, 0.8, 0.6, 0.05, 0.9])
        served_flagged = np.array([False, True, False, False, True])  # one miss, one FP

        metrics = evaluate_stored_decisions(1, y, amounts, scores, served_flagged, 5.0, "w")
        assert metrics.pr_auc == pytest.approx(average_precision_score(y, scores))
        # missed the 100.0 fraud, one false alert at review cost 5, over 5 rows
        assert metrics.cost_per_100k == pytest.approx((100.0 + 5.0) / 5 * 100_000)
        assert metrics.n_labelled == 5

    def test_no_positives_gives_nan_pr_auc_but_real_cost(self) -> None:
        from fraudops.models.window_eval import evaluate_stored_decisions

        y = np.array([0, 0, 0])
        metrics = evaluate_stored_decisions(
            1,
            y,
            np.array([5.0, 5.0, 5.0]),
            np.array([0.9, 0.1, 0.2]),
            np.array([True, False, False]),
            5.0,
            "w",
        )
        assert np.isnan(metrics.pr_auc)  # undefined — the gate rejects NaN
        assert metrics.cost_per_100k == pytest.approx(5.0 / 3 * 100_000)


def test_make_training_frame_columns() -> None:
    # guard: the window tests rely on conftest's frame carrying the pipeline's
    # required columns; fail loudly here if the fixture ever changes shape
    frame = make_training_frame()
    for col in ("TransactionID", "TransactionDT", "TransactionAmt", "isFraud"):
        assert col in frame.columns
