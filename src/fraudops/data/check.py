"""Dataset presence and shape validation (``make data``).

Checks the two raw files exist, carry the expected schema and row counts.
Purely structural — no labels or statistics are inspected here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

EXPECTED_ROWS = {
    "train_transaction.csv": 590_540,
    "train_identity.csv": 144_233,
}
REQUIRED_COLUMNS = {
    "train_transaction.csv": ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt"],
    "train_identity.csv": ["TransactionID"],
}


def _count_data_rows(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(buf.count(b"\n") for buf in iter(lambda: fh.read(1 << 20), b"")) - 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    ok = True
    for name, expected in EXPECTED_ROWS.items():
        path = args.data_dir / name
        if not path.exists():
            print(f"MISSING  {path} — download steps: README, Dataset section")
            ok = False
            continue
        header = path.open(encoding="utf-8").readline().strip().split(",")
        missing_cols = [c for c in REQUIRED_COLUMNS[name] if c not in header]
        rows = _count_data_rows(path)
        size_mb = path.stat().st_size / 1e6
        row_status = "ok" if rows == expected else f"EXPECTED {expected}"
        col_status = "ok" if not missing_cols else f"MISSING {missing_cols}"
        print(f"{path}  {rows:,} rows [{row_status}]  {size_mb:,.0f} MB  columns [{col_status}]")
        ok = ok and rows == expected and not missing_cols

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
