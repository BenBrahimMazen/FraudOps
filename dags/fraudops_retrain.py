"""Closed loop: drift trigger -> retrain on labelled data -> gated promotion.

``check_trigger`` re-derives the retrain decision from the monitor's persisted
evidence (the same pure ``should_retrain`` the monitor evaluates each cycle)
and skips the run when nothing fired. The challenger trains on the original
split plus released labels, and is promoted ONLY through the cost/PR-AUC gate
— every decision lands in ``promotion_log`` and MLflow. After a promotion the
API and scorer pick the new champion up through their alias pollers
(zero-downtime reload, no restart task needed).

No task here loads a serving bundle: reference numbers come from MLflow run
metrics, the champion's gate metrics from its stored predictions, and the
challenger is the one model this container itself logged (the orchestrator's
constrained python env is not the env that pickled earlier champions — see
docs/decisions.md).

Schedule is deliberately manual (``schedule=None``): the drift reference is
the champion's original training distribution, so an injection that stays
armed keeps firing the trigger — an hourly schedule would retrain on every
wake-up. The monitor exposes the trigger state as a Prometheus gauge for an
external scheduler; see docs/decisions.md.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException

DATA_DIR = Path(os.environ.get("FRAUDOPS_DATA_DIR", "/data/raw"))
CONFIGS_DIR = Path("/opt/airflow/configs")
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://fraudops:fraudops-local@postgres:5432/fraudops"
)


@dag(
    dag_id="fraudops_retrain",
    schedule=None,
    start_date=datetime(2026, 9, 29, tzinfo=UTC),
    catchup=False,
    tags=["fraudops", "closed-loop"],
)
def fraudops_retrain():
    @task
    def check_trigger() -> dict:
        import psycopg
        import yaml

        from fraudops.monitoring.drift import DriftResult, Level
        from fraudops.monitoring.trigger import TriggerConfig, should_retrain
        from fraudops.registry import client as rc

        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]

        # Reference = the champion's training distribution, taken from the
        # metrics its own run logged (train_* for the original split,
        # retrain_val_* for a champion that was itself retrained on labels).
        # Deliberately NOT a model load: the orchestrator's python env is not
        # the env that pickled the serving bundles (Airflow constraints keep
        # numpy older than the lock), and unpickling there fails — the DAG
        # reads recorded numbers instead.
        cli = rc.configure(tracking_uri)
        champion_version = rc.get_version_by_alias(cli, rc.CHAMPION)
        if champion_version is None:
            raise AirflowSkipException("no champion registered — run bootstrap first")
        run = cli.get_run(rc.alias_run_id(cli, rc.CHAMPION))
        metrics = run.data.metrics

        def _ref(*keys: str) -> float | None:
            for key in keys:
                if key in metrics:
                    return metrics[key]
            return None

        reference_pr_auc = _ref("train_pr_auc", "retrain_val_pr_auc")
        reference_cost = _ref("train_cost_per_100k", "retrain_val_cost_per_100k")

        with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
            # one monitor cycle writes ~22 rows; the last 60 rows bracket it
            cur.execute(
                """
                    SELECT kind, name, psi, ks_pvalue, level, details
                    FROM monitoring_results
                    WHERE id > (SELECT MAX(id) - 60 FROM monitoring_results)
                    ORDER BY id
                    """
            )
            rows = cur.fetchall()

        score_rows = [r for r in rows if r[0] == "score"]
        if not score_rows:
            raise AirflowSkipException("no monitoring evidence yet — run the monitor")
        latest_feature: dict[str, tuple] = {}
        for row in rows:
            if row[0] == "feature":
                latest_feature[row[1]] = row
        score = DriftResult(
            name="score",
            psi=score_rows[-1][2],
            ks_pvalue=score_rows[-1][3],
            level=Level(score_rows[-1][4]),
        )
        features = [
            DriftResult(name=name, psi=row[2], ks_pvalue=row[3], level=Level(row[4]))
            for name, row in latest_feature.items()
        ]

        perf = None
        perf_rows = [r for r in rows if r[0] == "performance"]
        if perf_rows:
            details = perf_rows[-1][5] or {}
            from fraudops.monitoring.trigger import PerformanceSnapshot

            perf = PerformanceSnapshot(
                pr_auc=details.get("pr_auc"),
                cost_per_100k=float(details.get("cost_per_100k", 0.0)),
                n_labelled=int(details.get("n_labelled", 0)),
            )

        with (CONFIGS_DIR / "drift.yaml").open(encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        decision = should_retrain(
            score,
            features,
            perf,
            reference_pr_auc=reference_pr_auc,
            reference_cost_per_100k=reference_cost,
            config=TriggerConfig(**cfg.get("trigger", {})),
        )
        if not decision.retrain:
            raise AirflowSkipException(
                f"trigger quiet: score PSI {score.psi:.3f}, "
                f"{len(features)} features, perf "
                f"{'n/a' if perf is None else f'pr_auc {perf.pr_auc}'}"
            )
        print("RETRAIN TRIGGERED: " + "; ".join(decision.reasons))
        return {
            "reasons": decision.reasons,
            "champion_version": int(champion_version),
            "reference_pr_auc": reference_pr_auc,
            "reference_cost_per_100k": reference_cost,
        }

    @task
    def retrain(evidence: dict) -> dict:
        import yaml

        from fraudops.models.retrain import train_challenger

        with (CONFIGS_DIR / "drift.yaml").open(encoding="utf-8") as fh:
            cfg = (yaml.safe_load(fh) or {}).get("retrain") or {}
        out = train_challenger(
            database_dsn=DATABASE_URL,
            tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
            data_dir=DATA_DIR,
            configs_dir=CONFIGS_DIR,
            window_sim_days=int(cfg.get("window_sim_days", 7)),
            val_days=int(cfg.get("val_days", 7)),
        )
        return {
            "run_id": out["run_id"],
            "threshold": out["threshold"],
            "gate_eval": {k: round(v, 6) for k, v in out["metrics"]["gate_eval"].items()},
        }

    @task
    def register_challenger(info: dict) -> int:
        from fraudops.registry import client as rc

        cli = rc.configure(os.environ["MLFLOW_TRACKING_URI"])
        version = rc.register_run_version(cli, info["run_id"])
        rc.set_alias(cli, rc.CHALLENGER, version)
        print(
            f"registered challenger v{version} (gate_eval pr_auc="
            f"{info['gate_eval']['pr_auc']}, threshold={info['threshold']:.4f})"
        )
        return int(version)

    @task
    def promotion_gate(challenger_version: int) -> dict:
        import yaml

        from fraudops.models.window_eval import compare_on_window
        from fraudops.registry import client as rc
        from fraudops.registry.gate import DEFAULT_GATE_CONFIG, GateConfig, apply_gate

        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        model_name = os.environ.get("FRAUDOPS_MODEL_NAME", "fraudops-lightgbm")
        cli = rc.configure(tracking_uri)
        champion_version = int(rc.get_version_by_alias(cli, rc.CHAMPION))

        with (CONFIGS_DIR / "costs.yaml").open(encoding="utf-8") as fh:
            false_alert_cost = float((yaml.safe_load(fh) or {}).get("false_alert_cost", 5.0))
        with (CONFIGS_DIR / "drift.yaml").open(encoding="utf-8") as fh:
            drift_cfg = yaml.safe_load(fh) or {}
        gate_cfg = drift_cfg.get("gate") or {}
        window_sim_days = int(drift_cfg.get("retrain", {}).get("window_sim_days", 7))

        champion, challenger = compare_on_window(
            database_dsn=DATABASE_URL,
            tracking_uri=tracking_uri,
            model_name=model_name,
            data_dir=DATA_DIR,
            false_alert_cost=false_alert_cost,
            champion_version=champion_version,
            challenger_version=challenger_version,
            window_sim_days=window_sim_days,
        )
        decision = apply_gate(
            cli,
            DATABASE_URL,
            champion,
            challenger,
            GateConfig(**gate_cfg) if gate_cfg else DEFAULT_GATE_CONFIG,
        )
        if decision.promote:
            print(
                "champion alias moved — API and scorer reload through their "
                "alias pollers (no restart needed)"
            )
        return {"promote": decision.promote, "reason": decision.reason}

    promotion_gate(register_challenger(retrain(check_trigger())))


fraudops_retrain()
