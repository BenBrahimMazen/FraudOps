"""Pydantic v2 API schemas, generated from the single feature schema.

Field definitions are derived from ``fraudops.features.schema`` so the API
contract can never drift from the feature pipeline: every input field the
pipeline expects exists here, and vice versa. Only the three identity/time/
amount fields are required; everything else may be absent (missing = NaN, a
signal the model consumes natively).
"""

from __future__ import annotations

from typing import Any

import pydantic
from pydantic import Field

from fraudops.data import loader
from fraudops.features import schema as fs

_REQUIRED_INT = (int, ...)
_REQUIRED_FLOAT = (float, ...)


def _categorical_annotation(col: str) -> tuple:
    """API type for a categorical feature, following the loader's storage
    classification: string-stored columns accept strings; numeric-valued
    ID-like columns (card2/3/5, addr1/2...) accept numbers."""
    string_stored = set(loader._TX_CATEGORY_COLUMNS + loader._ID_CATEGORY_COLUMNS)
    if col == "card1":
        return (int | None, None)  # integer storage
    if col in string_stored:
        return (str | None, None)
    return (float | None, None)


def _build_transaction_model() -> type[pydantic.BaseModel]:
    fields: dict[str, Any] = {
        "TransactionID": _REQUIRED_INT,
        "TransactionDT": _REQUIRED_INT,
        "TransactionAmt": _REQUIRED_FLOAT,
    }
    for col in fs.NUMERIC_TX + fs.NUMERIC_ID:
        if col == "TransactionAmt":
            continue
        fields[col] = (float | None, None)
    for col in fs.CATEGORICAL_TX + fs.CATEGORICAL_ID:
        fields[col] = _categorical_annotation(col)
    return pydantic.create_model("Transaction", **fields)


# Typed Any on purpose: the classes are generated at import from the feature
# schema, so FastAPI sees the real model at runtime while type checkers treat
# request bodies as untyped (the API tests pin the actual contract).
Transaction: Any = _build_transaction_model()

MAX_BATCH = 1000

BatchRequest: Any = pydantic.create_model(
    "BatchRequest",
    transactions=(list[Transaction], Field(..., max_length=MAX_BATCH)),
)

ScoreResponse = pydantic.create_model(
    "ScoreResponse",
    fraud_probability=(float, ...),
    decision=(str, ...),
    threshold=(float, ...),
    model_version=(int, ...),
    top_reasons=(list[dict], ...),
    latency_ms=(float, ...),
)
