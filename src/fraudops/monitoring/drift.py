"""Population Stability Index and KS drift statistics.

Pure functions with known-answer tests. Bins are fitted on the TRAINING
reference window and reused for every later window — refitting bins on the
drifted data would hide the drift.

PSI interpretation (configurable thresholds, defaults):
    psi <= 0.1        ok
    0.1 < psi <= 0.2  warning
    psi > 0.2         alert
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy import stats

DEFAULT_WARNING = 0.1
DEFAULT_ALERT = 0.2


class Level(StrEnum):
    OK = "ok"
    WARNING = "warning"
    ALERT = "alert"


@dataclass(frozen=True)
class DriftResult:
    name: str
    psi: float
    ks_pvalue: float | None
    level: Level

    @property
    def is_alert(self) -> bool:
        return self.level is Level.ALERT


def level_for(psi: float, warning: float = DEFAULT_WARNING, alert: float = DEFAULT_ALERT) -> Level:
    if psi > alert:
        return Level.ALERT
    if psi > warning:
        return Level.WARNING
    return Level.OK


# --------------------------------------------------------------------- PSI
def fit_numeric_bins(reference: np.ndarray, bins: int = 10) -> np.ndarray:
    """Quantile bin edges fitted on the reference sample (inclusive range)."""
    reference = np.asarray(reference, dtype="float64")
    reference = reference[np.isfinite(reference)]
    if reference.size == 0:
        raise ValueError("reference sample is empty")
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    # quantile edges can collide for skewed data; make them strictly increasing
    edges = np.unique(edges)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _bin_shares(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total == 0:
        return np.full_like(counts, 1.0 / max(len(counts), 1), dtype="float64")
    return counts / total


def psi_numeric(
    reference: np.ndarray,
    actual: np.ndarray,
    edges: np.ndarray | None = None,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """PSI between two numeric samples using reference-fitted quantile bins."""
    reference = np.asarray(reference, dtype="float64")
    actual = np.asarray(actual, dtype="float64")
    if reference.size == 0 or actual.size == 0:
        raise ValueError("psi needs non-empty reference and actual samples")
    if edges is None:
        edges = fit_numeric_bins(reference, bins=bins)
    ref_counts = np.histogram(reference, bins=edges)[0]
    act_counts = np.histogram(actual, bins=edges)[0]
    p = np.clip(_bin_shares(ref_counts), epsilon, None)
    q = np.clip(_bin_shares(act_counts), epsilon, None)
    return float(np.sum((q - p) * np.log(q / p)))


def psi_categorical(reference: np.ndarray, actual: np.ndarray, epsilon: float = 1e-6) -> float:
    """PSI for categorical data: one bin per reference category (plus an
    'unseen' bin for actual values outside the reference vocabulary)."""
    reference = np.asarray(reference, dtype=object)
    actual = np.asarray(actual, dtype=object)
    if reference.size == 0 or actual.size == 0:
        raise ValueError("psi needs non-empty reference and actual samples")
    categories = sorted({v for v in reference if v is not None and not _is_nan(v)})
    ref_counts = np.array([int(np.sum(reference == c)) for c in categories] + [0], dtype="float64")
    unseen = ~np.isin(actual, categories)
    act_counts = np.array(
        [int(np.sum(actual == c)) for c in categories] + [int(unseen.sum())],
        dtype="float64",
    )
    p = np.clip(_bin_shares(ref_counts), epsilon, None)
    q = np.clip(_bin_shares(act_counts), epsilon, None)
    return float(np.sum((q - p) * np.log(q / p)))


def psi(reference: np.ndarray, actual: np.ndarray, **kwargs) -> float:
    """PSI for numeric or categorical data (categorical when non-numeric)."""
    sample = np.asarray(reference)
    if sample.dtype == object or sample.dtype.kind in "USb":
        return psi_categorical(reference, actual)
    return psi_numeric(reference, actual, **kwargs)


def _is_nan(value) -> bool:
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


# ---------------------------------------------------------------------- KS
def ks_pvalue(reference: np.ndarray, actual: np.ndarray) -> float | None:
    """Two-sample KS test p-value (second opinion for PSI; None if unusable)."""
    reference = np.asarray(reference, dtype="float64")
    actual = np.asarray(actual, dtype="float64")
    reference = reference[np.isfinite(reference)]
    actual = actual[np.isfinite(actual)]
    if reference.size < 2 or actual.size < 2:
        return None
    return float(stats.ks_2samp(reference, actual).pvalue)


def evaluate_drift(
    name: str,
    reference: np.ndarray,
    actual: np.ndarray,
    warning: float = DEFAULT_WARNING,
    alert: float = DEFAULT_ALERT,
    **psi_kwargs,
) -> DriftResult:
    """PSI + KS on one feature (numeric or categorical)."""
    sample = np.asarray(reference)
    if sample.dtype == object or sample.dtype.kind in "USb":
        psi_value = psi_categorical(reference, actual)
        ks = None
    else:
        psi_value = psi_numeric(reference, actual, **psi_kwargs)
        ks = ks_pvalue(reference, actual)
    return DriftResult(
        name=name,
        psi=psi_value,
        ks_pvalue=ks,
        level=level_for(psi_value, warning, alert),
    )
