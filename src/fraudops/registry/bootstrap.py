"""Registry bootstrap and operations CLI.

Actions:
  train-champion   train a model, register it, set BOTH challenger and
                   champion aliases (first-model bootstrap)
  status           alias map + recent versions
  promote          move the champion alias to the challenger's version
  rollback [N]     restore the previous champion (or version N)

Runs anywhere with the fraudops package: locally against the SQLite store or
inside the stack against the MLflow server (MLFLOW_TRACKING_URI).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from fraudops.registry import client


def _client_from_env() -> client.MlflowClient:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "./mlruns/mlflow.db")
    if uri.startswith("sqlite:///"):
        raw = uri.removeprefix("sqlite:///")
        if raw.startswith("./"):  # default local store: ensure the directory exists
            Path(raw).parent.mkdir(parents=True, exist_ok=True)
    return client.configure(uri)


def cmd_train_champion(args: argparse.Namespace) -> None:
    from fraudops.models.train import train_baseline

    cli = _client_from_env()
    out = train_baseline(
        configs_dir=Path(args.configs_dir),
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        reports_dir=Path(args.reports_dir),
    )
    version = client.register_run_version(cli, out["run_id"])
    client.set_alias(cli, client.CHALLENGER, version)
    client.set_alias(cli, client.CHAMPION, version)
    print(
        f"registered version {version} (run {out['run_id']}) as "
        f"{client.CHALLENGER} and {client.CHAMPION} "
        f"(threshold {out['threshold']:.4f})"
    )


def cmd_status(_: argparse.Namespace) -> None:
    info = client.status(_client_from_env())
    print(f"model: {info['model_name']}")
    for alias, version in info["aliases"].items():
        print(f"  @{alias:<10} version {version}")
    print("recent versions:")
    for v in info["versions"]:
        marker = " <- champion" if str(v["version"]) == info["aliases"]["champion"] else ""
        print(f"  v{v['version']}  run {v['run_id'][:8]}...{marker}")


def cmd_promote(_: argparse.Namespace) -> None:
    result = client.promote_challenger(_client_from_env())
    print(f"promoted challenger to champion: {result}")


def cmd_rollback(args: argparse.Namespace) -> None:
    result = client.rollback_champion(_client_from_env(), args.to)
    print(f"champion rolled back: {result}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    p = sub.add_parser("train-champion", help="train, register, set champion (first model)")
    p.add_argument("--configs-dir", type=Path, default=Path("configs"))
    p.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--out-dir", type=Path, default=Path("models"))
    p.add_argument("--reports-dir", type=Path, default=Path("reports"))
    p.set_defaults(func=cmd_train_champion)

    sub.add_parser("status", help="aliases + versions").set_defaults(func=cmd_status)
    sub.add_parser("promote", help="challenger -> champion").set_defaults(func=cmd_promote)

    p = sub.add_parser("rollback", help="restore previous champion (or version N)")
    p.add_argument("to", nargs="?", default=None)
    p.set_defaults(func=cmd_rollback)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
