"""Simulated clock derived from TransactionDT.

TransactionDT is a time offset in seconds of unknown absolute origin, spanning
about 182 days. All temporal logic in FraudOps uses this offset directly; the
display anchor (default 2017-12-01, a common convention for this dataset) is
cosmetic only and never enters features or splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86_400

# Cosmetic anchor for human-readable dates only. Ordering logic never uses it.
DEFAULT_ANCHOR = "2017-12-01"


def sim_day(transaction_dt: pd.Series | np.ndarray) -> np.ndarray:
    """Simulated day index (0-based) for a TransactionDT offset in seconds."""
    return np.asarray(transaction_dt) // SECONDS_PER_DAY


def hour_of_day(transaction_dt: pd.Series | np.ndarray) -> np.ndarray:
    """Hour of the simulated day (0-23). Cyclical, relative — anchor-independent."""
    return (np.asarray(transaction_dt) // 3600) % 24


def day_of_week(transaction_dt: pd.Series | np.ndarray) -> np.ndarray:
    """Day of the simulated week (0-6). Cyclical, relative — anchor-independent."""
    return (np.asarray(transaction_dt) // SECONDS_PER_DAY) % 7
