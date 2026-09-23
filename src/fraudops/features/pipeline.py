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

# frames at or below this size take the values-first assembly path (serving);
# larger frames (training/evaluation) use the vectorised path
_SMALL_FRAME_ROWS = 32


class FeaturePipeline:
    """Fit on training rows; transform any transaction frame to model features."""

    def __init__(self, use_v_columns: bool = False) -> None:
        self.use_v_columns = use_v_columns
        self.categories_: dict[str, pd.Index] = {}
        self.numeric_columns_: list[str] = []
        self.v_columns_: list[str] = []
        self.feature_names_: list[str] = []
        # derived caches (rebuilt on load via __setstate__)
        self._cat_dtypes_: dict[str, pd.CategoricalDtype] = {}
        self._cat_sets_: dict[str, frozenset] = {}
        self._cat_maps_: dict[str, dict] = {}

    def __setstate__(self, state: dict) -> None:
        # joblib/pickle restore: rebuild the derived caches from categories_
        self.__dict__.update(state)
        self._rebuild_caches()

    def _rebuild_caches(self) -> None:
        """Per-column lookups derived from categories_.

        The serving path transforms one row at a time and must not rebuild
        categorical machinery over the full vocabulary per request.
        """
        self._cat_dtypes_ = {
            col: pd.CategoricalDtype(categories=cats) for col, cats in self.categories_.items()
        }
        self._cat_sets_ = {col: frozenset(cats) for col, cats in self.categories_.items()}
        self._cat_maps_ = {
            col: {value: code for code, value in enumerate(cats)}
            for col, cats in self.categories_.items()
        }

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
        self._rebuild_caches()
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the model matrix with columns in ``feature_names_`` order.

        Same input + same fitted state always produce the same output —
        training and serving share this exact method. Small frames (the
        serving path) take a values-first assembly that avoids per-column
        pandas machinery; branch parity is unit-tested.
        """
        if len(df) <= _SMALL_FRAME_ROWS:
            return self._transform_small(df)
        return self._transform_vectorised(df)

    def _transform_vectorised(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._derive_has_identity(df)

        # Engineered features (anchor-independent, cyclical where temporal).
        engineered = pd.DataFrame(
            {
                "log_TransactionAmt": np.log1p(
                    pd.to_numeric(df["TransactionAmt"], errors="coerce").astype("float32")
                ),
                "sim_hour": hour_of_day(df["TransactionDT"]).astype("float32"),
                "sim_dow": day_of_week(df["TransactionDT"]).astype("float32"),
            },
            index=df.index,
        )

        # Numeric block in one assignment (columns are typed upstream on both
        # the training and serving paths; missing columns raise here).
        num_cols = self.numeric_columns_ + self.v_columns_
        missing_cols = [c for c in num_cols if c not in df.columns]
        if missing_cols:
            raise KeyError(f"features missing after transform: {missing_cols}")
        numeric = df[num_cols].astype("float32")

        # Categoricals: encode via the cached value->code maps and build the
        # Categorical from codes (O(1); astype(CategoricalDtype) would rebuild
        # over the full category list on every call). Unseen values map to
        # code -1 == missing, exactly like the training-side semantics.
        cat_parts = []
        for col, mapping in self._cat_maps_.items():
            if col not in df.columns:
                continue  # vocabulary learned without this column
            codes = df[col].map(mapping).fillna(-1).astype("int32")
            cat_parts.append(
                pd.Series(
                    pd.Categorical.from_codes(codes.to_numpy(), dtype=self._cat_dtypes_[col]),
                    index=df.index,
                    name=col,
                )
            )

        out = pd.concat([engineered, numeric, *cat_parts], axis=1)
        missing = [c for c in self.feature_names_ if c not in out.columns]
        if missing:
            raise KeyError(f"features missing after transform: {missing}")
        return out.loc[:, self.feature_names_]

    def _transform_small(self, df: pd.DataFrame) -> pd.DataFrame:
        """Values-first assembly for few-row frames (serving path).

        Extracts the object matrix in one pass and builds typed arrays
        directly — identical semantics to the vectorised branch, without
        per-column pandas machinery. Branch parity is unit-tested.
        """
        n = len(df)
        columns = list(df.columns)
        pos = {c: i for i, c in enumerate(columns)}
        matrix = df.to_numpy() if columns else np.empty((n, 0), dtype=object)

        num_cols = self.numeric_columns_ + self.v_columns_

        def column(col: str) -> list:
            i = pos[col]
            return [matrix[r, i] for r in range(n)]

        def _missing(value) -> bool:
            return value is None or (isinstance(value, float) and np.isnan(value))

        data: dict[str, np.ndarray | pd.Categorical] = {}

        # has_identity derivation for serving frames that lack the loader flag
        # (same semantics as the vectorised branch: any identity column set)
        if "has_identity" in num_cols and "has_identity" not in pos:
            id_cols = [c for c in schema.CATEGORICAL_ID + schema.NUMERIC_ID if c in pos]
            data["has_identity"] = np.array(
                [not all(_missing(matrix[r, pos[c]]) for c in id_cols) for r in range(n)],
                dtype="float32",
            )

        missing_cols = [c for c in num_cols if c not in pos and c not in data]
        if missing_cols:
            raise KeyError(f"features missing after transform: {missing_cols}")
        amt = np.array(
            [np.nan if _missing(v) else v for v in column("TransactionAmt")], dtype="float32"
        )
        data["log_TransactionAmt"] = np.log1p(amt)
        dt = np.array(column("TransactionDT"), dtype="int64")
        data["sim_hour"] = ((dt // 3600) % 24).astype("float32")
        data["sim_dow"] = ((dt // 86400) % 7).astype("float32")
        for col in num_cols:
            if col in data:
                continue  # derived above (has_identity)
            data[col] = np.array(
                [np.nan if _missing(v) else v for v in column(col)], dtype="float32"
            )
        for col, mapping in self._cat_maps_.items():
            if col not in pos:
                continue  # vocabulary learned without this column
            codes = np.array(
                [-1 if _missing(v) else mapping.get(v, -1) for v in column(col)],
                dtype="int32",
            )
            data[col] = pd.Categorical.from_codes(codes, dtype=self._cat_dtypes_[col])

        out = pd.DataFrame(data, index=df.index)
        missing = [c for c in self.feature_names_ if c not in out.columns]
        if missing:
            raise KeyError(f"features missing after transform: {missing}")
        return out.loc[:, self.feature_names_]

    def _derive_has_identity(self, df: pd.DataFrame) -> pd.DataFrame:
        # Serving frames arrive without the loader's has_identity flag:
        # derive it here with identical semantics (any identity column set),
        # so training and serving construct the same feature.
        if "has_identity" in self.numeric_columns_ and "has_identity" not in df.columns:
            id_cols = [c for c in schema.CATEGORICAL_ID + schema.NUMERIC_ID if c in df.columns]
            df = df.assign(has_identity=df[id_cols].notna().any(axis=1))
        return df

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
