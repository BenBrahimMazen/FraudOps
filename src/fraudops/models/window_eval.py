"""Closed-loop window evaluation: champion vs challenger on real labelled data.

The gate needs both models' metrics on the **most recent fully-labelled
window** — the transactions whose labels the release job has already made
visible. Two fidelity rules make the comparison honest:

- Features are rebuilt from the raw CSVs and the ACTIVE drift injections are
  re-applied, so the window looks exactly like the stream the models served
  (a drifted ``TransactionAmt`` stays drifted; the champion is not quietly
  handed a cleaner test set than it faced in production).
- Truth and cost amounts come from the ``labels`` table (staged from the
  pre-injection row): the injected shift corrupts what the model sees, never
  ground truth.

Each model scores the window through its OWN pipeline at its OWN shipped
threshold — the deployment configuration, not a threshold re-tuned on the
window being judged (that would flatter whichever model overfits the window).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraudops.data.clock import sim_day
from fraudops.data.loader import load_joined
from fraudops.models.cost import cost_per_100k
from fraudops.registry.gate import WindowMetrics
from fraudops.registry.wrapper import FraudOpsModel

SECONDS_PER_DAY = 86_400


# ------------------------------------------------------------- drift replay


def fetch_active_injections(conn) -> list[dict]:
    """Active injections from Postgres (same rows the replay producer reads)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT kind, params, from_sim_day FROM drift_injections WHERE active ORDER BY id"
        )
        return [
            {"kind": kind, "params": params, "from_day": from_day}
            for kind, params, from_day in cur.fetchall()
        ]


def apply_injections_frame(df: pd.DataFrame, injections: list[dict]) -> pd.DataFrame:
    """Vectorised twin of the replay ``Injector``: the same synthetic shifts,
    applied to a frame, for days at or past each injection's ``from_day``."""
    df = df.copy()
    if not injections:
        return df
    day = df["TransactionDT"] // SECONDS_PER_DAY
    for injection in injections:
        mask = day >= int(injection["from_day"])
        if not mask.any():
            continue
        if injection["kind"] == "amount_factor":
            factor = float(injection["params"].get("factor", 1.0))
            df.loc[mask, "TransactionAmt"] = df.loc[mask, "TransactionAmt"] * factor
        elif injection["kind"] == "nullify":
            for col in injection["params"].get("columns", []):
                if col not in df.columns:
                    continue
                # categorical columns cannot take .loc-assigned NaN: widen to
                # object first — transform() maps missing to -1 either way
                if str(df[col].dtype) == "category":
                    df[col] = df[col].astype("object")
                df.loc[mask, col] = np.nan
    return df


# ------------------------------------------------------------ window fetch


def fetch_labelled_window(
    conn,
    data_dir: Path,
    window_sim_days: int = 7,
    injections: list[dict] | None = None,
    min_rows: int = 500,
) -> tuple[pd.DataFrame, dict]:
    """Raw features + labels for the most recent fully-labelled window.

    Window boundaries follow label AVAILABILITY (``available_at_dt``), so by
    construction every transaction inside is released — the same rule the
    label-release job enforces. Raises when the window is too small to gate on.
    Returns (frame, info) with the boundary seconds in ``info`` so callers can
    query the exact same window from other tables.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(MAX(available_at_dt), 0) FROM labels")
        cutoff = cur.fetchone()[0]
        if not cutoff:
            raise RuntimeError("no labels released yet — nothing to evaluate on")
        start = cutoff - window_sim_days * SECONDS_PER_DAY
        cur.execute(
            """
            SELECT transaction_id, is_fraud, amount
            FROM labels
            WHERE available_at_dt > %s AND available_at_dt <= %s
            """,
            (start, cutoff),
        )
        rows = cur.fetchall()
    if len(rows) < min_rows:
        raise RuntimeError(
            f"fully-labelled window too small to gate on: {len(rows)} rows < {min_rows}"
        )

    truth = pd.DataFrame(rows, columns=["TransactionID", "is_fraud", "amount"])
    df = load_joined(data_dir / "train_transaction.csv", data_dir / "train_identity.csv")
    window = df[df["TransactionID"].isin(truth["TransactionID"])].merge(
        truth, on="TransactionID", how="inner", validate="one_to_one"
    )
    # features reflect the stream (injections re-applied); truth stays clean
    if injections is None:
        injections = fetch_active_injections(conn)
    window = apply_injections_frame(window, injections)
    window["is_fraud"] = window["is_fraud"].astype("int8")

    first_day = sim_day(window["TransactionDT"]).min()
    last_day = sim_day(window["TransactionDT"]).max()
    info = {
        "start": int(start),
        "cutoff": int(cutoff),
        "desc": f"sim days {first_day}-{last_day} ({len(window):,} labelled)",
    }
    return window, info


# ---------------------------------------------------------------- scoring


def load_bundle(tracking_uri: str, model_name: str, version: int) -> FraudOpsModel:
    """Load one registered version (aliases are for serving; the gate compares
    explicit versions so a promotion cannot race the comparison)."""
    import mlflow

    pyfunc = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
    return FraudOpsModel.unwrap(pyfunc)


def score_frame(bundle: FraudOpsModel, frame: pd.DataFrame) -> np.ndarray:
    """Score through the bundle's own pipeline — the exact serving path."""
    return bundle.score_raw(frame)


def evaluate_on_window(
    bundle: FraudOpsModel,
    version: int,
    frame: pd.DataFrame,
    false_alert_cost: float,
    window_desc: str,
) -> WindowMetrics:
    """PR-AUC + cost per 100k for one model on the labelled window, at its
    shipped threshold (no re-tuning on the judged window)."""
    y = frame["is_fraud"].to_numpy()
    amounts = frame["amount"].to_numpy(dtype="float64")
    scores = score_frame(bundle, frame)
    flagged = scores >= bundle.threshold
    pr_auc = float(average_precision_score(y, scores)) if y.sum() > 0 else float("nan")
    return WindowMetrics(
        version=version,
        pr_auc=pr_auc,
        cost_per_100k=cost_per_100k(y, amounts, flagged, false_alert_cost),
        n_labelled=len(frame),
        window_desc=window_desc,
        threshold=float(bundle.threshold),
    )


def evaluate_stored_decisions(
    version: int,
    y: np.ndarray,
    amounts: np.ndarray,
    scores: np.ndarray,
    flagged: np.ndarray,
    false_alert_cost: float,
    window_desc: str,
) -> WindowMetrics:
    """Window metrics from what a model ACTUALLY served.

    The champion is judged on its stored production predictions — scores and
    decisions as they left the scorer — not on a re-score. (Two reasons: the
    served numbers are the deployment reality, and the orchestrator's python
    environment differs from the serving one that pickled the model, so
    re-scoring the champion there is not even possible.)
    """
    pr_auc = float(average_precision_score(y, scores)) if y.sum() > 0 else float("nan")
    return WindowMetrics(
        version=version,
        pr_auc=pr_auc,
        cost_per_100k=cost_per_100k(y, amounts, flagged, false_alert_cost),
        n_labelled=len(y),
        window_desc=window_desc,
    )


def champion_window_metrics(
    conn,
    version: int,
    start: int,
    cutoff: int,
    false_alert_cost: float,
    window_desc: str,
    expected_n: int,
) -> WindowMetrics:
    """The champion's metrics on the window, from its own stored predictions.

    ``expected_n`` is the labelled window size: if the champion did not score
    every window transaction (a serving gap, or a mid-window promotion), the
    comparison would be across different row sets — fail loudly instead.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT l.is_fraud, l.amount, p.fraud_probability, p.decision
            FROM labels l
            JOIN predictions p ON p.transaction_id = l.transaction_id
            WHERE l.available_at_dt > %s AND l.available_at_dt <= %s
              AND p.model_version = %s
            """,
            (start, cutoff, version),
        )
        rows = cur.fetchall()
    if len(rows) != expected_n:
        raise RuntimeError(
            f"champion v{version} scored {len(rows)} of {expected_n} window "
            f"transactions — mixed-version window, refusing to compare"
        )
    y = np.array([r[0] for r in rows], dtype=int)
    amounts = np.array([r[1] for r in rows], dtype="float64")
    scores = np.array([r[2] for r in rows], dtype="float64")
    flagged = np.array([r[3] == "alert" for r in rows], dtype=bool)
    return evaluate_stored_decisions(
        version, y, amounts, scores, flagged, false_alert_cost, window_desc
    )


def compare_on_window(
    database_dsn: str,
    tracking_uri: str,
    model_name: str,
    data_dir: Path,
    false_alert_cost: float,
    champion_version: int,
    challenger_version: int,
    window_sim_days: int = 7,
) -> tuple[WindowMetrics, WindowMetrics]:
    """Both models' metrics on the most recent fully-labelled window.

    Champion: its own stored predictions (what it served). Challenger: the
    freshly-registered bundle, scored through its own pipeline on the same
    window, features rebuilt with the live drift injections applied.
    """
    import psycopg

    with psycopg.connect(database_dsn) as conn:
        frame, info = fetch_labelled_window(conn, data_dir, window_sim_days)
        champion = champion_window_metrics(
            conn,
            champion_version,
            info["start"],
            info["cutoff"],
            false_alert_cost,
            info["desc"],
            len(frame),
        )

    challenger_bundle = load_bundle(tracking_uri, model_name, challenger_version)
    challenger = evaluate_on_window(
        challenger_bundle, challenger_version, frame, false_alert_cost, info["desc"]
    )
    print(
        f"window {info['desc']}: "
        f"champion v{champion_version} (served scores) pr_auc={champion.pr_auc:.3f} "
        f"cost/100k={champion.cost_per_100k:,.0f} | "
        f"challenger v{challenger_version} pr_auc={challenger.pr_auc:.3f} "
        f"cost/100k={challenger.cost_per_100k:,.0f}"
    )
    return champion, challenger
