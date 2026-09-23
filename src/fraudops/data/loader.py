"""Loading of the IEEE-CIS training files with memory-efficient dtypes.

Only the two labelled training files are used:

- ``train_transaction.csv`` (~590k rows)
- ``train_identity.csv``    (~144k rows, a subset of transactions)

joined left on TransactionID so no transaction row is lost; a ``has_identity``
flag marks the ~24% of rows carrying identity information.

Dtypes are declared up front (float32 for numerics, category for strings) to
keep the joined frame well under 1 GB instead of the ~2 GB a naive float64
read would need.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

# Integer columns (TransactionDT max ~15.7M fits int32 comfortably).
_INT_COLUMNS: dict[str, str] = {
    "TransactionID": "int64",
    "TransactionDT": "int32",
    "isFraud": "int8",
    "card1": "int32",
}

# String/categorical columns in the transaction file (T/F flags, labels, domains).
_TX_CATEGORY_COLUMNS = [
    "ProductCD",
    "card4",
    "card6",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "P_emaildomain",
    "R_emaildomain",
]

# String columns in the identity file (id_12..id_38 are labels, not numbers).
_ID_CATEGORY_COLUMNS = [
    "id_12",
    "id_15",
    "id_16",
    "id_23",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
    "DeviceType",
    "DeviceInfo",
]


def _read_csv(path: Path, category_columns: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    dtypes: dict[str, object] = {}
    for col in header.columns:
        if col in _INT_COLUMNS:
            dtypes[col] = _INT_COLUMNS[col]
        elif col in category_columns:
            dtypes[col] = "category"
        else:
            dtypes[col] = "float32"
    return pd.read_csv(path, dtype=dtypes)


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load train_transaction.csv with declared dtypes."""
    return _read_csv(Path(path), _TX_CATEGORY_COLUMNS)


def load_identity(path: str | Path) -> pd.DataFrame:
    """Load train_identity.csv with declared dtypes."""
    return _read_csv(Path(path), _ID_CATEGORY_COLUMNS)


def load_joined(transactions_path: str | Path, identity_path: str | Path) -> pd.DataFrame:
    """Left-join identity onto transactions and add a ``has_identity`` flag.

    Rows without identity information keep NaN in the id_*/Device* columns —
    missingness itself is informative and is never imputed. The flag marks
    rows that HAVE an identity record: some identity records carry NaN
    DeviceInfo, so the flag is computed as "any identity column non-null",
    not from a single column.
    """
    tx = load_transactions(transactions_path)
    ident = load_identity(identity_path)
    identity_cols = [c for c in ident.columns if c != "TransactionID"]
    with warnings.catch_warnings():
        # inserting one column into the freshly merged frame is fine; pandas's
        # fragmentation advice here would force a full copy of ~590k rows
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        joined = tx.merge(ident, on="TransactionID", how="left", validate="one_to_one")
        joined["has_identity"] = joined[identity_cols].notna().any(axis=1)
    return joined
