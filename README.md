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
| 3 | Kafka replay, delayed labels, drift monitoring | done |
| 4 | Closed loop: trigger → retrain → gate → promote/rollback | done |
| 5 | Experiments, load testing, figures | done |
| 6 | Terraform/LocalStack, CI/CD polish | pending |

Every metric below comes from a real run and can be regenerated with the
command shown; nothing is estimated.

## Quickstart (current state)

Requirements: Python 3.11, [`uv`](https://docs.astral.sh/uv/), GNU `make`
(on Windows: `winget install ezwinports.make`), Docker Desktop (Phases 2+).

```bash
uv sync        # or: make install — create the locked virtualenv
make lint      # ruff check + format check
make test      # pytest (140 tests: unit, leakage, parity, gate, integration)
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

Concurrent Locust traffic on the same stack: see the Phase 5 load test.

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

## Streaming, delayed labels and drift monitoring (Phase 3)

`make up-full` starts Kafka (KRaft, single node) plus the scorer consumer,
the label-release job, the drift monitor, Prometheus and Grafana. Then
`make drift-amount FACTOR=3 FROM_DAY=165` arms a synthetic shift and
`make replay DAY_SECONDS=8` replays the stream split through it:

- **Replay** (`make replay DAY_SECONDS=8`, 2026-09-28): 94,636 events
  (sim days 150–182) published in TransactionDT order at **328.6 events/s**.
- **Scoring**: the consumer scores every event with the champion through the
  same library the API uses (no serving-only code path) — **94,636 of 94,636
  predictions stored**, 94,636 distinct transaction ids, 1,892,720 feature
  rows (exactly the top-20 features per prediction), 28,508 alerts (30.1%) at
  the cost-optimal threshold 0.0296.
- **Delayed labels**: labels never travel on the topic. The producer stages
  them in `labels_pending`; the release job moves them to `labels` once the
  simulated clock passes `TransactionDT + 7 days`. Final state:
  76,759 released + 17,877 pending = 94,636 exactly.
- **Drift monitoring**: 87 cycles (15 s apart), PSI + KS on the top-20 gain
  features plus score PSI, trailing 7-sim-day window, every result in
  `monitoring_results` and Prometheus/Grafana (`fraudops_drift_psi` gauges).

**Injected shift, detected as designed.** With amounts ×3 from sim day 165:

| sim day | `log_TransactionAmt` PSI | level |
|---|---|---|
| 150–166 | 0.004 – 0.081 | ok |
| 167 | 0.188 | **warning** — 2 sim days after injection |
| 169 | 0.460 | **alert** — 4 sim days after |
| 170 → 182 | 0.93 → 1.61 | sustained alert |

**Score drift never fired**: score PSI stayed ≤ 0.039 the whole replay
(warning level is 0.1). The champion's probability distribution was stable
under both the ×3 amount shift and heavy natural feature drift — feature
drift ≠ score drift. The retrain trigger fires on feature PSI; Phase 4 gates
what actually happens next.

**Natural drift was alerting from day 150**: `id_31`/`id_30`/`id_33`
(browser/OS metadata, PSI up to 7.4), `DeviceInfo` (2.6), `R_emaildomain`
(2.3). The train window (days 0–119) and the stream differ enough in device
mix to trigger continuously against a fixed training reference — honest
behaviour, not a bug: population change is exactly what a fixed-reference PSI
is meant to flag.

Two data-loss bugs were found and fixed during live verification (the kind
of thing only a real end-to-end run surfaces):

1. **Kafka message timestamps** — the producer stamped messages with
   dataset-epoch *seconds* where Kafka expects epoch *milliseconds*: every
   message dated 1970-01-01, so segment roll timers and the 7-day retention
   both saw "56-year-old" data and the broker's 5-minute sweep deleted
   segments under the lagging scorer, twice (31,936 then 38,440 predictions
   silently skipped — offsets committed, lag 0, no errors anywhere). Fix: no
   message timestamps; simulated time travels in the payload, which the
   consumer, label release and monitor already read. Side effect: producer
   throughput went 188.7 → 328.6 events/s (no more micro-segment churn).
2. **Consumer tail freeze** — `consumer.poll(0.2)` once blocked 8+ minutes on
   a librdkafka-internal futex (SIGINT traceback pinned it), and the idle
   branch never flushed the write buffer, so the last 136 predictions reached
   Postgres only via the shutdown flush. Fix: idle-branch flush + a 5 s
   watchdog flusher thread; the scorer now restarts unless-stopped. A hung
   `poll()` still stalls consumption until restart — documented limitation.

`make report-evidently` writes an HTML drift report for the last window
(`reports/evidently/`), built against the same champion training reference
the live monitor uses.

## The closed loop (Phase 4)

With the ×3 amount shift still armed and the replay finished, the loop was
closed end to end on 2026-09-29 (DAG run
`manual__2026-09-29T09:47:58…IMjhkz5x`, all four tasks green):

1. **Trigger** — `check_trigger` re-derives the retrain decision from the
   monitor's persisted evidence using the same pure `should_retrain` the
   monitor evaluates each cycle. Evidence at decision time: 5 top-20 features
   above PSI 0.2 (worst: `id_31` 4.48, `log_TransactionAmt` 1.58), labelled
   PR-AUC 0.989 → **0.494** and cost 117,360 → **232,603** per 100k on the
   trailing labelled window; score PSI 0.019 (quiet, as in Phase 3).
2. **Retrain** — the challenger trains on the original train split **plus the
   released labelled stream** (injections re-applied, so it learns the drifted
   regime), with its cost-optimal threshold tuned on a chronological labelled
   slice just before the gate window — never on the window being judged.
3. **Gate** — both models on the most recent fully-labelled window
   (sim days 168–175, 19,943 transactions, features rebuilt with the live
   injection applied). The champion is judged on the scores and decisions
   **it actually served** (stored predictions joined to labels); the
   challenger through its own pipeline at its own shipped threshold.

| window sim days 168–175 (n = 19,943) | PR-AUC | cost / 100k |
|---|---|---|
| champion v1 (as served) | 0.494 | 232,603 |
| challenger v3 (retrained) | 0.556 | **182,935** |
| gate rule | PR-AUC must not drop | cost must improve ≥ 1% |

**Decision: promoted** — cost improved 21.4% with PR-AUC up. The decision and
both models' metrics are in the `promotion_log` table and in MLflow; the
champion alias moved v1 → v3 and the challenger alias was cleared
automatically.

4. **Zero-downtime reload** — the API and the scorer pick the new champion up
   through their alias pollers: `/model` showed v3 **16 seconds** after the
   gate decision, `reload_count` incremented, no restart. Steady-state
   `/score` latency on v3: **51–82 ms** over 10 requests (first request after
   a reload pays ~1.3 s of lazy warmup). The monitor rebuilt its drift
   reference for v3 on its next cycle.

`make rollback` (or `make rollback TO=1` for an explicit version) restores an
earlier champion through the same alias mechanism. The gate itself is a pure
function with table-driven tests — promote, reject, tie-within-margin,
PR-AUC-loss-blocks — plus registry-level tests for alias moves and rollback.

**Why the orchestrator never loads a serving model.** Airflow installs
fraudops under its own dependency constraints (numpy 1.x) while serving images
install the project lock (numpy 2.x); cloudpickle artifacts embed numpy
internals, so a serving-pickled model cannot be unpickled under Airflow
(verified the hard way: `ModuleNotFoundError: numpy._core.numeric`). The DAG
therefore reads the champion's reference metrics from its MLflow run, judges
the champion on its stored production predictions, and only loads the
challenger — the one model logged in that same container. Cross-checked: the
stored-prediction metrics matched an independent rescore from raw exactly
(PR-AUC 0.494, cost 232,603, n = 19,943). Pickles logged under Airflow load
fine in the serving images (the compatible direction), with loud but harmless
version-mismatch warnings.

Two monitor bugs were found and fixed while wiring the gate (regression-tested
in `tests/test_monitor.py`):

1. **The labelled-performance set was empty by construction** — with a 7-day
   window and a 7-day label delay, a prediction becomes labelable exactly when
   it ages out of the window, so the labelled query returned zero rows and
   performance monitoring silently never ran (the trigger had been firing on
   feature PSI alone). The lookback now spans window + delay; the released-at
   filter still decides visibility.
2. **psycopg refuses plain dicts at JSONB placeholders** — every monitor cycle
   aborted at insert once performance details were added. Wrapped in
   `Json(...)`.

The DAG is deliberately manual (`schedule=None`): the drift reference is the
champion's original training distribution, so an injection that stays armed
keeps the trigger hot — a schedule would retrain on every wake-up. The monitor
exposes the trigger state as the `fraudops_drift_trigger` Prometheus gauge for
an external scheduler; Phase 5 measures detection lag off exactly this signal.

## Experiments and figures (Phase 5, one command: `make report`)

Every figure and table regenerates from the raw data plus the live
databases — `make report` runs all five report scripts (each also works
standalone as `uv run python -m fraudops.reports.<name>`), and every
experiment arm is logged to the MLflow experiment `fraudops_experiments`
so each number traces to a run. All measurements on this machine: AMD
Ryzen 5 3500U (4 cores / 8 threads, laptop), 16 GB RAM, Docker Desktop
VM ≈ 8.8 GB, Windows 10, API container without CPU/memory limits.

### 1. Frozen-model decay — `reports/figures/decay_curve.{csv,png}`

Champion v1 judged only on what it actually served: stored
scores/decisions joined to the labels released 7 sim days later,
trailing 7-day windows stepped one day (the monitor's own window shape).

PR-AUC erodes from 0.554 (window ending day 157) to a 0.420 trough
(day 170), recovering to 0.494 by day 175 — always below the 0.595
validation reference, i.e. the model arrived already degraded and kept
sliding. Expected cost per 100k climbs 208k → 283k (peak day 173) →
233k while the alert rate creeps 26.8% → 33.5%: the model drifts toward
crying wolf, each false alert still costing 5.0. The ×3 injection
(day 165) shows up more sharply in cost than PR-AUC — tripling amounts
triples the price of every miss without moving ranking quality much,
and 7-day windows straddle the boundary so nothing is a step change.

### 2. Retraining policies — `reports/figures/retrain_comparison.{csv,png}`

Three policies over the labelled replay (sim days 150–175, 76,649
transactions). Each retrain rebuilds exactly what that moment could know
(`as_of` label visibility — the same rule the release job enforces) and
fits through the same `fit_challenger` the live DAG uses; the no-retraining
arm is the frozen champion's actual served record.

| policy | retrains | total cost 150–175 | active window | cost/100k (active) | PR-AUC (active) |
|---|---|---|---|---|---|
| no retraining (v1 as served) | 0 | 174,355 | — | 227,676 (whole period) | 0.494 |
| scheduled weekly | 1 — at day 171 | 171,340 | 171–175 | 180,905 | 0.538 |
| drift-triggered (PSI alert) | 1 — at day 169 | 169,412 | 169–175 | 207,855 | 0.519 |

The 7-day label delay dominates the whole comparison: the ×3 shift lands
at day 165 but its **labels only exist from day 172**, so neither retrain
could train on the drift itself — both improve through fresher
natural-drift labels and a re-tuned threshold (0.081 / 0.115 vs v1's
0.0296, i.e. far fewer false alerts once amounts are tripled). The weekly
schedule **starves** before it starts: ticks at day 157 (no labels
visible) and day 164 (the 7 visible days are entirely consumed by
threshold validation, leaving zero training rows) are skipped exactly as
a real scheduler would — its first usable tick is day 171, and a 14-day
cadence never gets a usable window at all inside this horizon.

The triggered arm wins on total cost (169.4k vs 171.3k) because it takes
over two days earlier and covers the champion's two worst decay days
(169–170, windows of 271k/280k per 100k) with a fresh model — not
because it knows more about the drift. The scheduled arm's model is
actually better per transaction on its own window (180.9k/100k, PR-AUC
0.538, trained with two more label days), but that is outweighed by
serving decayed v1 through days 169–170. The active windows differ
(5 vs 7 days), so the per-100k columns are not head-to-head. The best
model of all was the live loop's v3 — trained once the drifted labels
existed (cutoff day 168): 182.9k/100k at PR-AUC 0.556 on the gate
window, the Phase 4 promotion. Every arm run is logged to MLflow
(`fraudops_experiments`). Honest caveat: single seed, single replay —
the 2–3% margins between arms are within plausible seed variance.

### 3. Threshold sensitivity to review cost — `reports/figures/threshold_sensitivity.{csv,png}`

The shipped threshold (review cost 5.0) lands at 0.029609 — exactly the
baseline's shipped 0.0296, reproduced by the same `optimal_cost_threshold`
the trainer uses: the figure is the training rule, not a re-derivation.

| review cost | optimal threshold | alert rate | recall | cost /100k |
|---|---|---|---|---|
| 1 | 0.0083 | 42.5% | 94.7% | 65,696 |
| 5 | 0.0296 | 26.7% | 89.2% | 188,073 |
| 10 | 0.1514 | 10.8% | 74.0% | 253,740 |
| 25 | 0.4708 | 4.0% | 59.6% | 314,625 |

The operating point is extremely sensitive to the review cost — a 25×
change moves the threshold 56× and halves recall. Missed fraud stays the
dominant cost share everywhere (37–85%), so even at review cost 25 the
optimum is not "alert on nothing". The realistic-operational-loads
observation from Phase 1 stands: at review cost 5 (average fraud ≈ 149)
the optimum tolerates a 27% alert rate; teams with human review
capacity would sit nearer the cost-10 row.

### 4. Detection lag by injected shift — `reports/figures/detection_lag.{csv,png}`

Six scenarios replayed **offline through the exact monitor semantics**
(v1's reference, top-20 features, PSI warn 0.1 / alert 0.2, trailing
7-day windows), each with only its own injection applied:

| scenario | warning lag | alert lag | max PSI |
|---|---|---|---|
| amount ×1.5 | 3 d | 4 d | 0.54 |
| amount ×2 | 3 d | 4 d | 0.66 |
| amount ×3 (the live one) | 2 d | 2 d | 1.60 |
| amount ×5 | 1 d | 2 d | 2.82 |
| nullify C13 | 6 d | 6 d | 0.31 |
| nullify P_emaildomain | masked | masked | 25.76 |

Bigger shifts are caught faster (×5 warns within a day; ×1.5 takes
three), and subtle corruption (a counting feature nulled) takes ~6 days
— PSI needs the window to fill with shifted data before the histogram
moves. The ×3 offline row (2-day alert) is a lower bound on the live
measurement (warning day 167, alert day 169): offline windows are
complete simulated days, while the live monitor evaluates partial
windows as the replay clock moves. `nullify P_emaildomain` is
**masked**: that channel was already in natural-drift alert before the
injection (first warning day 157 < 165), so this shift's own lag is not
measurable — a real caveat of fixed-reference drift monitoring, shown
rather than hidden. Score PSI never crossed 0.1 in any scenario (max
0.054 at ×5), consistent with Phase 3: feature drift ≠ score drift.

### 5. Load test — Locust on `POST /score`

Saturation profile (no think time), full stack running (12 containers
incl. monitor and Airflow), API in Docker without CPU/memory limits,
**single Uvicorn worker**, Locust on the host. Hardware: AMD Ryzen 5
3500U (4C/8T laptop), 16 GB RAM, Docker Desktop VM ≈ 8.8 GB, Windows 10.
Headline run: `make loadtest USERS=50 RUN_TIME=2m`; the 1/10/25-user
points come from the same file with `-u` overridden.

| concurrent users | requests | throughput | p50 | p95 | p99 | failures |
|---|---|---|---|---|---|---|
| 1 | 675 | 15.0 /s | 60 ms | **90 ms** | 140 ms | 0 |
| 10 | 882 | 14.8 /s | 650 ms | 960 ms | 1.0 s | 0 |
| 25 | 776 | 13.2 /s | 1.8 s | 2.7 s | 3.0 s | 0 |
| 50 | 1,362 | 11.5 /s | 3.7 s | 5.9 s | 13.0 s | 0 |

Reading it honestly: the **p95 < 100 ms target is met for a single
concurrent request** (90 ms p95, 60 ms p50 — matching the Phase 2
steady-state numbers), and the service never errors under saturation —
but it saturates at **~13–15 requests/s** regardless of offered
concurrency, with latency growing linearly in the queue (Little's law
holds at every level: users ≈ rps × latency). The cause is structural:
one Uvicorn worker, and a `/score` path dominated by GIL-bound Python
(Pydantic validation, DataFrame build, pandas feature transform) around
a native TreeSHAP call that releases the GIL for only part of the work —
so the process serialises at roughly one request per single-request
latency. Four cores cannot help one Python process. What would move it:
multiple Uvicorn workers (one model copy each — a memory cost on this
stack), moving the hot path off pandas, or leaning on `/score/batch`
(~4 ms/row amortised). Documented as the deployment's real capacity
envelope: correct and predictable, but single-process.

### 6. SHAP summary and reason codes — `reports/figures/shap_summary.png`, `reason_codes.md`

Champion v3 explained on a 5,000-transaction labelled sample (sim days
168–175, seed 42). Three example `/score` outputs, top-3 features by
|contribution| exactly as the API computes them:

| case | score | decision | top reasons (log-odds) |
|---|---|---|---|
| highest score | 0.999 | alert (actual fraud) | `C1`=5.0 +2.06, `C14`=0 +1.51, `C13`=0 +1.24 |
| borderline | 0.084 | approve (actual legit) | `card1`=11298 +1.21, `C5`=1 −0.69, `log_TransactionAmt` +0.64 |
| lowest score | 0.000 | approve (actual legit) | `DeviceInfo` −2.14, `card1` −1.60, `card3` −1.35 |

The beeswarm shows anonymised counters (`C1`, `C13`, `C14`, `C5`) and
`card1`/`card3` doing most of the work, with amount contributing
positively. Reason codes are technical, not analyst narratives — an
honest consequence of the anonymised schema (see Limitations), and the
borderline case shows why: `card1` pushing +1.2 toward fraud with no
business gloss a reviewer could act on.

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
- **Serving is single-process**: one Uvicorn worker meets the p95 < 100 ms
  target per request at low concurrency but saturates at ~13–15 req/s
  under load (GIL-bound scoring path; measured envelope in Phase 5).
  Multiple workers or a non-pandas hot path would raise the ceiling at
  the cost of one model copy per process.
- **Airflow runs as a single container** (`standalone`, LocalExecutor) — a
  local-scale deployment, not a production Airflow topology; the fraudops
  package is installed under Airflow's constraint set, so its pandas/fastapi
  versions differ from the application lock file inside that one image.
- **The full stack + retraining brushes the Docker-VM memory ceiling** (~8.6 GB
  allocated): the retrain task peaks near 2 GB on top of Airflow, the monitor
  (~2 GB) and the rest of the stack. One DAG run was killed by the kernel OOM
  killer during task spawn before the monitor was stopped for the duration of
  retraining — at real scale this is a capacity-planning number, not a
  workaround. The one closed-loop run shown above completed with ~1 GB spare.
- **Cross-environment model artifacts**: models logged under Airflow's
  constraints load in the serving images with loud version-mismatch warnings
  (cloudpickle/sklearn/numpy), not silently; the reverse direction fails
  outright, which is why no Airflow task ever loads a serving model.
- **One promotion is demonstrated**, not a long series: the loop's rejection
  and tie paths are covered by unit tests on the pure gate function, but the
  live run shown promoted on its first decision.

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
