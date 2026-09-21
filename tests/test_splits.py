"""Chronological split and leakage-guard tests (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from fraudops.data.splits import (
    STREAM,
    TRAIN,
    VALIDATION,
    assert_chronological,
    chronological_split,
)

SEC_PER_DAY = 86_400


def make_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt = np.sort(rng.integers(0, 200 * SEC_PER_DAY, size=n))
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": dt,
            "isFraud": rng.integers(0, 2, size=n),
            "TransactionAmt": rng.uniform(1, 500, size=n),
        }
    )


class TestChronologicalSplit:
    def test_partition_is_exact_and_disjoint(self) -> None:
        df = make_frame()
        splits = chronological_split(df, train_days=100, val_days=50)
        total = sum(len(p) for p in splits.values())
        assert total == len(df)
        ids = pd.concat([p["TransactionID"] for p in splits.values()])
        assert ids.is_unique

    def test_train_precedes_validation_precedes_stream(self) -> None:
        splits = chronological_split(make_frame(), train_days=100, val_days=50)
        assert splits[TRAIN]["TransactionDT"].max() < splits[VALIDATION]["TransactionDT"].min()
        assert splits[VALIDATION]["TransactionDT"].max() < splits[STREAM]["TransactionDT"].min()

    def test_boundaries_follow_configured_days(self) -> None:
        df = make_frame()
        splits = chronological_split(df, train_days=100, val_days=50)
        assert splits[TRAIN]["TransactionDT"].max() < 100 * SEC_PER_DAY
        assert splits[VALIDATION]["TransactionDT"].min() >= 100 * SEC_PER_DAY
        assert splits[VALIDATION]["TransactionDT"].max() < 150 * SEC_PER_DAY
        assert splits[STREAM]["TransactionDT"].min() >= 150 * SEC_PER_DAY

    def test_rejects_non_positive_windows(self) -> None:
        with pytest.raises(ValueError):
            chronological_split(make_frame(), train_days=0, val_days=50)

    def test_rejects_empty_split(self) -> None:
        short = make_frame(n=10)
        # 10 rows all within days 0..~5 -> stream window empty
        with pytest.raises(ValueError, match="empty split"):
            chronological_split(short, train_days=3, val_days=3)


class TestLeakageGuard:
    def test_passes_on_ordered_splits(self) -> None:
        splits = chronological_split(make_frame(), train_days=100, val_days=50)
        assert_chronological(splits)  # must not raise

    def test_fails_when_train_overlaps_validation(self) -> None:
        splits = chronological_split(make_frame(), train_days=100, val_days=50)
        leaked = splits[TRAIN].copy()
        leaked["TransactionDT"] = leaked["TransactionDT"] + 200 * SEC_PER_DAY
        with pytest.raises(AssertionError, match="train"):
            assert_chronological({**splits, TRAIN: leaked})

    def test_fails_when_validation_overlaps_stream(self) -> None:
        splits = chronological_split(make_frame(), train_days=100, val_days=50)
        leaked = splits[VALIDATION].copy()
        leaked["TransactionDT"] = leaked["TransactionDT"] + 200 * SEC_PER_DAY
        with pytest.raises(AssertionError, match="validation"):
            assert_chronological({**splits, VALIDATION: leaked})
