# FraudOps

Real-time fraud detection with a **closed-loop MLOps pipeline**: a chronological
replay of the IEEE-CIS card-transaction dataset streams through Kafka, is
scored in real time by a FastAPI service (LightGBM + SHAP reason codes), and is
monitored for feature/score/performance drift (custom PSI/KS + Evidently,
Prometheus/Grafana). When drift triggers retraining, a challenger is promoted
to champion **only** if it beats the incumbent on a cost-based gate — with a
logged, reversible decision and a rollback path.

Portfolio project. Everything runs locally with Docker Compose (MinIO for
S3-compatible storage, LocalStack for AWS emulation) — zero cloud cost.
Engineering rigor over model novelty: the point is what happens *after* a
model is deployed.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Scaffolding, tooling, CI | done |
| 1 | Data, chronological split, LightGBM baseline, cost threshold | pending |
| 2 | MLflow/MinIO registry, Airflow, FastAPI serving | pending |
| 3 | Kafka replay, delayed labels, drift monitoring | pending |
| 4 | Closed loop: trigger → retrain → gate → promote/rollback | pending |
| 5 | Experiments, load testing, figures | pending |
| 6 | Terraform/LocalStack, CI/CD polish | pending |

No results exist yet — every metric on this README will say **"not yet
measured"** until a real pipeline run produces it, and each figure will come
with the command that regenerates it.

## Quickstart (current state)

Requirements: Python 3.11, [`uv`](https://docs.astral.sh/uv/), GNU `make`
(on Windows: `winget install ezwinports.make`). Docker Desktop is needed from
Phase 2 onward.

```bash
uv sync        # or: make install — create the locked virtualenv
make lint      # ruff check + format check
make test      # pytest
```

The full stack (`make up`) arrives in Phase 2.

## Dataset

**IEEE-CIS Fraud Detection** (Kaggle competition data, governed by Kaggle's
rules — accepted per account, so the download is manual; `data/` is gitignored
and never committed).

1. Sign in at [kaggle.com](https://www.kaggle.com).
2. Accept the competition rules once:
   <https://www.kaggle.com/competitions/ieee-fraud-detection/rules>
3. Download the two labelled training files (links work in your browser while
   signed in and after accepting the rules):
   - [train_transaction.csv.zip](https://www.kaggle.com/api/v1/competitions/data/download/ieee-fraud-detection/train_transaction.csv.zip) (~139 MB)
   - [train_identity.csv.zip](https://www.kaggle.com/api/v1/competitions/data/download/ieee-fraud-detection/train_identity.csv.zip) (~27 MB)
4. Extract both so the layout is exactly:

   ```
   fraudops/data/raw/train_transaction.csv
   fraudops/data/raw/train_identity.csv
   ```

Expected: ~590,540 data rows in `train_transaction.csv`, ~144,233 in
`train_identity.csv`. Only these two training files are used; the
competition's `test_*` files carry no labels and are never used for
evaluation.

## Limitations

Maintained honestly from day one (every claim must trace to a run):

- **Replay, not production traffic.** The stream is a chronological replay of
  a public 2019 dataset; volumes, arrival patterns and fraud behaviour are
  simulated, and the speed-up factor distorts time further.
- **Anonymised features.** Most IEEE-CIS columns (V1–V339, C*, D*, M*) carry
  no business meaning; SHAP reason codes will be technical, not analyst
  narratives.
- **No measured performance yet.** PR-AUC, cost per 100k transactions, p95
  latency: all "not yet measured" until the relevant phase runs.
- **Single machine.** All latency/load numbers will be laptop-specific
  (hardware documented alongside each figure).
- **Data licensing.** Kaggle competition terms apply; this repo contains only
  download instructions, never data.
- **Not for real use.** Research/portfolio project, not intended for real
  financial decisions.
