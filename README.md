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
| 1 | Data, chronological split, LightGBM baseline, cost threshold | done |
| 2 | MLflow/MinIO registry, Airflow, FastAPI serving | done |
| 3 | Kafka replay, delayed labels, drift monitoring | pending |
| 4 | Closed loop: trigger → retrain → gate → promote/rollback | pending |
| 5 | Experiments, load testing, figures | pending |
| 6 | Terraform/LocalStack, CI/CD polish | pending |

Every metric below comes from a real run and can be regenerated with the
command shown; nothing is estimated.

## Quickstart (current state)

Requirements: Python 3.11, [`uv`](https://docs.astral.sh/uv/), GNU `make`
(on Windows: `winget install ezwinports.make`), Docker Desktop (Phases 2+).

```bash
uv sync        # or: make install — create the locked virtualenv
make lint      # ruff check + format check
make test      # pytest (65 fast tests + real-data guard)
make data      # validate the two CSVs in data/raw/
make train     # local run: load -> split -> features -> LightGBM -> threshold -> evaluation -> MLflow
make up        # core stack: postgres, minio, mlflow, api
make bootstrap # train + register + set the first champion (inside the stack)
make up-full   # + airflow (single container, LocalExecutor, Postgres metadata)
```

Exploration notebook: `notebooks/01_eda.ipynb` (executed outputs committed).

## The serving stack (Phase 2)

`make up` starts the `core` profile — Postgres (predictions + MLflow +
Airflow databases), MinIO (S3-compatible artifacts), an MLflow server
(Postgres backend, MinIO artifacts) and the scoring API. `make up-full` adds
Airflow (`fraudops_train` DAG: train → evaluate → register as challenger).

**Registry.** The model is ONE MLflow pyfunc artifact carrying the model,
the feature pipeline and the cost-optimal threshold together — training and
serving cannot disagree about the decision rule. Aliases: `champion` (what
serving loads, `models:/fraudops-lightgbm@champion`) and `challenger` (what
training registers). `make status` / `make promote` / `make rollback`
operate the aliases.

**API** (`:8000`, contract in the OpenAPI docs at `/docs`):

| endpoint | behaviour |
|---|---|
| `POST /score` | probability, decision, threshold, model version, top-3 SHAP reasons, latency |
| `POST /score/batch` | up to 1,000 transactions |
| `GET /health`, `GET /model` | liveness; champion version/threshold/training window/metrics |
| `GET /metrics` | Prometheus: request counts, latency histograms, decisions, score distribution |
| `POST /admin/reload` | force champion reload (token header) |

Zero-downtime reload: a background poller swaps the champion atomically when
the alias changes (≤15 s), or `/admin/reload` forces it immediately. Every
scored transaction is persisted to Postgres through a bounded queue and a
batch writer — scoring never blocks on the database (drops are counted, not
raised). Serving 503s with a clear error while no champion exists.

**Measured on this machine** (single-user, 60 steady-state `/score` calls,
Docker stack, champion v1 — 500 trees, 96 features):

- scoring latency **p50 60.4 ms / p95 92.7 ms** (target: p95 < 100 ms;
  the p50 includes probability AND SHAP reasons from one fused
  LightGBM `pred_contrib` call)
- `/score/batch` of 100 real stream transactions: 73 approve / 27 alert
  (27% alert rate, matching the offline evaluation), ~4 ms/row amortised
- 288 predictions persisted to Postgres with model version + latency;
  `/metrics` exposes the same counts

The load-test numbers under concurrent Locust traffic arrive in Phase 5.

## Orchestration (Phase 2, `make up-full`)

A single-container Airflow (standalone, LocalExecutor, Postgres metadata)
runs the `fraudops_train` DAG: **train → evaluate → register challenger**,
reusing the exact same library the CLI bootstrap uses (no orchestration-only
code path). Measured end-to-end on this machine:

- DAG run `manual__2026-09-24T13:16:52Z`: both tasks green; training inside
  the container took 54.4 s; the run registered **version 2** as challenger
  with the same threshold as version 1 (0.02961 — same seed, same data,
  reproducible)
- `make promote` → champion 1→2; the serving API swapped models through its
  background poller **without a restart** (`/health` reported version 2
  within 20 s)
- `make rollback` → champion 2→1; the API followed back automatically

The Airflow image installs the package under Airflow's constraint set
(pandas 2.1.4 there vs 3.0.6 in the lock file); a container-side smoke
script (`scripts/smoke_airflow_env.py`) exercises both feature-pipeline
branches plus a LightGBM fit on that stack before DAG runs are trusted.

## Phase 1 results — baseline (one command: `make train`)

Chronological split in simulated days (0–119 / 120–149 / 150–182): train
410,601 rows (fraud rate 3.51%), validation 85,303 (3.47%), stream 94,636
(3.47%). The feature pipeline is fitted on the training split only; the
cost-optimal threshold (0.0296) is selected on validation only; the stream is
scored without ever being used for tuning.

| window | PR-AUC | ROC-AUC | recall @1% FPR | model cost /100k | no-model /100k | fixed 0.5 /100k | F1-optimal /100k |
|---|---|---|---|---|---|---|---|
| train | 0.988 | 0.999 | 0.993 | 117,737 | 510,174 | 8,377 | 28,199 |
| validation | 0.595 | 0.918 | 0.528 | **188,073** | 578,902 | 282,208 | 340,315 |
| stream | 0.500 | 0.884 | 0.442 | **211,523** | 523,315 | 325,717 | 352,308 |

(costs in the transaction currency; a missed fraud costs its amount, a false
alert costs 5.0 — `configs/costs.yaml`)

Honest reading:

- The train/validation PR-AUC gap (0.988 → 0.595) is deliberate overfitting
  left unpruned: no early stopping, because validation is reserved for
  threshold selection. Nothing looks suspiciously good on unseen data, and the
  chronological leakage guards all pass.
- The cost-optimal threshold beats all three baselines on validation and on
  the stream (211.5k vs 325.7k at fixed 0.5 vs 352.3k at the F1-optimal
  threshold vs 523.3k with no model).
- At a 5.0 review cost the optimum alerts ~27% of transactions — mathematically
  correct (average fraud amount ≈ 149) but operationally unrealistic; the
  threshold-sensitivity experiment (Phase 5) quantifies this trade-off.
- Degradation from validation to stream (PR-AUC 0.595 → 0.500) is the natural
  drift the monitoring phase will track and the retraining loop will act on.

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

- **Baseline overfits by design** (0.988 train vs 0.595 validation PR-AUC): no
  early stopping or hyperparameter search, to keep validation reserved for
  threshold selection and the stream untouched by tuning.
- **Feature set is deliberately small**: no anonymised V-columns and no
  competition-style group-key aggregation features; leaderboard-grade PR-AUC
  is explicitly not the goal of this project.
- **Latency numbers are single-user** so far: the p50/p95 above are
  steady-state sequential calls inside Docker on one laptop; concurrent
  Locust numbers (Phase 5) are not yet measured.
- **Airflow runs as a single container** (`standalone`, LocalExecutor) — a
  local-scale deployment, not a production Airflow topology; the fraudops
  package is installed under Airflow's constraint set, so its pandas/fastapi
  versions differ from the application lock file inside that one image.

Maintained honestly from day one (every claim must trace to a run):

- **Replay, not production traffic.** The stream is a chronological replay of
  a public 2019 dataset; volumes, arrival patterns and fraud behaviour are
  simulated, and the speed-up factor distorts time further.
- **Anonymised features.** Most IEEE-CIS columns (V1–V339, C*, D*, M*) carry
  no business meaning; SHAP reason codes will be technical, not analyst
  narratives.
- **Single machine.** All latency/load numbers will be laptop-specific
  (hardware documented alongside each figure).
- **Data licensing.** Kaggle competition terms apply; this repo contains only
  download instructions, never data.
- **Not for real use.** Research/portfolio project, not intended for real
  financial decisions.
