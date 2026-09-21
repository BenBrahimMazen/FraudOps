"""Chronological splitting driven by configs/splits.yaml.

Splits are computed on simulated-day boundaries so that every training
timestamp strictly precedes every validation timestamp, which strictly precedes
every stream timestamp. Random splits leak the future and are never used.
"""

from __future__ import annotations

import pandas as pd

from fraudops.data.clock import sim_day

TRAIN = "train"
VALIDATION = "validation"
STREAM = "stream"


def chronological_split(
    df: pd.DataFrame, train_days: int, val_days: int
) -> dict[str, pd.DataFrame]:
    """Split ``df`` by simulated day: [0, train_days) / [train_days, +val_days) / rest."""
    if train_days <= 0 or val_days <= 0:
        raise ValueError("train_days and val_days must be positive")
    day = sim_day(df["TransactionDT"])
    splits = {
        TRAIN: df[day < train_days].copy(),
        VALIDATION: df[(day >= train_days) & (day < train_days + val_days)].copy(),
        STREAM: df[day >= train_days + val_days].copy(),
    }
    counts = {name: len(part) for name, part in splits.items()}
    if sum(counts.values()) != len(df):
        raise AssertionError(f"split lost or duplicated rows: {counts} vs {len(df)}")
    if any(n == 0 for n in counts.values()):
        raise ValueError(f"empty split — check train_days/val_days: {counts}")
    assert_chronological(splits)
    return splits


def assert_chronological(splits: dict[str, pd.DataFrame]) -> None:
    """Leakage guard: train precedes validation precedes stream, strictly."""
    if TRAIN not in splits or VALIDATION not in splits or STREAM not in splits:
        raise KeyError(f"splits must contain {TRAIN}, {VALIDATION} and {STREAM}")
    max_train_dt = splits[TRAIN]["TransactionDT"].max()
    min_val_dt = splits[VALIDATION]["TransactionDT"].min()
    max_val_dt = splits[VALIDATION]["TransactionDT"].max()
    min_stream_dt = splits[STREAM]["TransactionDT"].min()
    if not max_train_dt < min_val_dt:
        raise AssertionError(
            f"leakage: train TransactionDT max {max_train_dt} >= validation min {min_val_dt}"
        )
    if not max_val_dt < min_stream_dt:
        raise AssertionError(
            f"leakage: validation TransactionDT max {max_val_dt} >= stream min {min_stream_dt}"
        )
