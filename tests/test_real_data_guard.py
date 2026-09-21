"""Leakage guard on the real dataset (skipped when data is absent, e.g. CI).

Kept separate from the fast unit tests: reading the raw CSV takes tens of
seconds, so this module is marked slow-but-local.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from fraudops.data.splits import chronological_split

DATA_FILE = Path("data/raw/train_transaction.csv")
SPLIT_CONFIG = Path("configs/splits.yaml")

pytestmark = pytest.mark.skipif(
    not DATA_FILE.exists(), reason="dataset not present (download: README, Dataset)"
)


def test_real_split_is_chronological() -> None:
    cfg = yaml.safe_load(SPLIT_CONFIG.read_text(encoding="utf-8"))
    # one column only: fast enough for a guard test
    dt = pd.read_csv(DATA_FILE, usecols=["TransactionDT"])
    splits = chronological_split(dt, int(cfg["train_days"]), int(cfg["val_days"]))
    # chronological_split() runs assert_chronological internally and raises on
    # any train/validation/stream overlap; partition exactness is also checked
    # inside. Reaching the row-count assertion means chronology holds.
    assert sum(len(p) for p in splits.values()) == len(dt)
