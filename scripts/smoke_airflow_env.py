"""Smoke-check the training-path stack inside the Airflow image.

Runs IN the airflow container (pandas under Airflow constraints, which
differs from the application lock file). Verifies the vectorised transform
branch end-to-end before any DAG run:

    docker compose --profile full exec -T airflow \
        python /opt/airflow/scripts/smoke_airflow_env.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudops.data.clock import sim_day
from fraudops.features.pipeline import FeaturePipeline


def main() -> None:
    print(f"pandas {pd.__version__}, numpy {np.__version__}")
    rng = np.random.default_rng(0)
    n = 300
    frame = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": rng.integers(0, 30 * 86_400, size=n),
            "TransactionAmt": rng.uniform(1, 500, size=n),
            "ProductCD": rng.choice(["W", "H"], size=n),
            "C1": rng.uniform(0, 5, size=n),
            "D1": rng.uniform(0, 20, size=n),
            "DeviceInfo": rng.choice(["Windows", "iOS"], size=n),
            "has_identity": rng.integers(0, 2, size=n).astype(bool),
        }
    )
    # loader-style dtypes: strings as category
    for col in ("ProductCD", "DeviceInfo"):
        frame[col] = frame[col].astype("category")

    pipe = FeaturePipeline()
    X = pipe.fit_transform(frame)  # vectorised branch (n > 32)
    assert X.shape[0] == n
    assert not X["ProductCD"].isna().any()
    X_small = pipe.transform(frame.head(2))  # serving branch
    for col in ("log_TransactionAmt", "C1"):
        assert np.isclose(X[col].iloc[0], X_small[col].iloc[0])

    # unseen category -> missing code
    later = frame.head(3).copy()
    later["ProductCD"] = pd.Categorical(["X_new", "W", "H"], categories=["X_new", "W", "H"])
    out = pipe.transform(later)
    assert pd.isna(out["ProductCD"].iloc[0])

    print("smoke OK: vectorised + small branches, unseen -> missing")


if __name__ == "__main__":
    main()
