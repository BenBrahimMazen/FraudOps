"""MLflow pyfunc wrapper: one artifact carrying model + pipeline + threshold.

The cost-optimal threshold travels WITH the model (as an attribute and in
metadata), so training and serving can never disagree about the decision
rule. ``predict`` returns P(fraud); the decision is threshold >= in the
serving layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel


class FraudOpsModel(PythonModel):
    """Scores raw transaction rows (same schema the feature pipeline expects)."""

    def __init__(
        self,
        model,  # LightGBM classifier (anything with predict_proba)
        pipeline,  # fraudops.features.pipeline.FeaturePipeline (fitted)
        threshold: float,
        metadata: dict | None = None,
    ) -> None:
        self.model = model
        self.pipeline = pipeline
        self.threshold = float(threshold)
        self.metadata = dict(metadata or {})

    def score_with_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Feature matrix and fraud probabilities for raw transaction rows.

        Uses the raw booster (predict for binary objective == predict_proba's
        positive column, without the sklearn wrapper's per-call overhead).
        """
        X = self.pipeline.transform(df)
        return X, self.model.booster_.predict(X)

    def score_raw(self, df: pd.DataFrame) -> np.ndarray:
        return self.score_with_features(df)[1]

    def decide(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores) >= self.threshold

    # --------------------------------------------------------- pyfunc contract
    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        """MLflow pyfunc contract: P(fraud) per row."""
        return self.score_raw(model_input)

    @staticmethod
    def unwrap(pyfunc_model) -> FraudOpsModel:
        """Recover this class from a loaded pyfunc model."""
        inner = pyfunc_model.unwrap_python_model()
        if not isinstance(inner, FraudOpsModel):
            raise TypeError("loaded model does not wrap a FraudOpsModel")
        return inner
