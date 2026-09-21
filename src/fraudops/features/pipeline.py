"""The one feature pipeline used everywhere (training, replay scoring, serving).

Fitted on training data only: categorical vocabularies are learned from the
training split; values unseen at transform time map to NaN (missing), which
LightGBM handles natively. Numeric missing values pass through as NaN — no
imputation. The fitted object is plain Python state serialized with joblib, so
the exact same transformer (and therefore the exact same features) is loaded
by every consumer.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraudops.data.clock import day_of_week, hour_of_day
from fraudops.features import schema


class FeaturePipeline:
    """Fit on training rows; transform any transaction frame to model features."""

    def __init__(self, use_v_columns: bool = False) -> None:
        self.use_v_columns = use_v_columns
        self.categories_: dict[str, pd.Index] = {}
        self.numeric_columns_: list[str] = []
        self.v_columns_: list[str] = []
        self.feature_names_: list[str] = []

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> FeaturePipeline:
        categorical = schema.CATEGORICAL_TX + schema.CATEGORICAL_ID
        for col in categorical:
            if col in df.columns:
                values = df[col].dropna().unique()
                self.categories_[col] = pd.Index(sorted(values))
        numeric_all = schema.NUMERIC_TX + schema.NUMERIC_ID + schema.FLAG_COLUMNS
        self.numeric_columns_ = [c for c in numeric_all if c in df.columns]
        if self.use_v_columns:
            self.v_columns_ = sorted(c for c in df.columns if c.startswith(schema.V_PREFIX))
        self.feature_names_ = self._column_order()
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the model matrix with columns in ``feature_names_`` order.

        Same input + same fitted state always produce the same output —
        training and serving share this exact method.
        """
        out = pd.DataFrame(index=df.index)

        # Engineered features (anchor-independent, cyclical where temporal).
        out["log_TransactionAmt"] = np.log1p(
            pd.to_numeric(df["TransactionAmt"], errors="coerce").astype("float32")
        )
        out["sim_hour"] = hour_of_day(df["TransactionDT"]).astype("float32")
        out["sim_dow"] = day_of_week(df["TransactionDT"]).astype("float32")

        for col in self.numeric_columns_ + self.v_columns_:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

        for col, categories in self.categories_.items():
            if col not in df.columns:
                continue  # categorical vocabulary learned without this column
            # Unseen-at-transform-time values map to missing (NaN): mask them
            # explicitly instead of relying on the deprecated Categorical
            # constructor behaviour.
            known = df[col].isin(categories)
            out[col] = df[col].where(known).astype(pd.CategoricalDtype(categories=categories))

        missing = [c for c in self.feature_names_ if c not in out.columns]
        if missing:
            raise KeyError(f"features missing after transform: {missing}")
        return out.loc[:, self.feature_names_]

    # ----------------------------------------------------------- persistence
    def _column_order(self) -> list[str]:
        return (
            schema.ENGINEERED
            + self.numeric_columns_
            + self.v_columns_
            + list(self.categories_.keys())
        )

    def save(self, path: str | Path) -> None:
        joblib.dump(self, Path(path))

    @staticmethod
    def load(path: str | Path) -> FeaturePipeline:
        obj = joblib.load(Path(path))
        if not isinstance(obj, FeaturePipeline):
            raise TypeError(f"{path} does not contain a FeaturePipeline")
        return obj
