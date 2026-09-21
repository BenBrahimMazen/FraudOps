"""Feature pipeline tests: train-only fitting, unseen handling, parity."""

import numpy as np
import pandas as pd

from fraudops.features.pipeline import FeaturePipeline


def make_frame(seed: int, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": rng.integers(0, 180 * 86_400, size=n),
            "TransactionAmt": rng.uniform(1, 1000, size=n),
            "ProductCD": rng.choice(["W", "H", "C"], size=n),
            "card4": rng.choice(["visa", "mastercard"], size=n),
            "C1": rng.uniform(0, 10, size=n),
            "D1": rng.uniform(0, 30, size=n),
            "id_02": rng.uniform(0, 100, size=n),
            "DeviceInfo": rng.choice(["Windows", "iOS", "Android"], size=n),
            "has_identity": rng.integers(0, 2, size=n).astype(bool),
        }
    )


def test_fit_learns_categories_from_train_only() -> None:
    train = make_frame(0)
    train = train.assign(ProductCD="W")  # single category in train
    pipe = FeaturePipeline().fit(train)
    assert list(pipe.categories_["ProductCD"]) == ["W"]
    # a category that exists only in a later window must not be in the vocabulary
    later = make_frame(1).assign(ProductCD="H")
    _ = pipe.transform(later)  # transforming a later window must not refit
    assert "H" not in pipe.categories_["ProductCD"]


def test_unseen_category_maps_to_missing() -> None:
    train = make_frame(0).assign(card4="visa")
    pipe = FeaturePipeline().fit(train)
    later = make_frame(1).assign(card4="mastercard")  # unseen at serve time
    out = pipe.transform(later)
    assert out["card4"].isna().all()
    # seen values keep their codes
    same = pipe.transform(train)
    assert same["card4"].notna().all()


def test_transform_is_deterministic_and_order_stable() -> None:
    train, probe = make_frame(0), make_frame(1)
    pipe = FeaturePipeline().fit(train)
    a, b = pipe.transform(probe), pipe.transform(probe)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == pipe.feature_names_


def test_save_load_roundtrip_produces_identical_features(tmp_path) -> None:
    train, probe = make_frame(0), make_frame(1)
    pipe = FeaturePipeline().fit(train)
    path = tmp_path / "pipeline.joblib"
    pipe.save(path)
    loaded = FeaturePipeline.load(path)
    pd.testing.assert_frame_equal(pipe.transform(probe), loaded.transform(probe))


def test_engineered_features_present() -> None:
    pipe = FeaturePipeline().fit(make_frame(0))
    out = pipe.transform(make_frame(1))
    for col in ("log_TransactionAmt", "sim_hour", "sim_dow"):
        assert col in out.columns
    assert np.isfinite(out["log_TransactionAmt"]).all()
    assert out["sim_hour"].between(0, 23).all()
    assert out["sim_dow"].between(0, 6).all()


def test_numeric_missingness_passes_through() -> None:
    frame = make_frame(0)
    frame.loc[0, "C1"] = np.nan
    pipe = FeaturePipeline().fit(frame)
    out = pipe.transform(frame)
    assert out.loc[0, "C1"] != out.loc[0, "C1"]  # NaN preserved, not imputed
