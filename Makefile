# FraudOps — one command per task.
# Targets marked [Phase N] are placeholders until that phase lands.

PY := uv run
USERS ?= 50
RUN_TIME ?= 2m
COMPOSE := docker compose
CORE := --profile core
FULL := --profile full
TOOLS := --profile tools

.DEFAULT_GOAL := help
.PHONY: help install lint typecheck format test precommit up down up-full bootstrap status \
        promote rollback logs data train replay drift loadtest report

help: ## List available targets
	@echo "FraudOps Makefile targets:"
	@echo "  install    - create/refresh the locked virtualenv (uv sync)"
	@echo "  lint       - ruff check + format check"
	@echo "  typecheck  - mypy over the package"
	@echo "  format     - apply ruff autofixes and formatting"
	@echo "  test       - run the pytest suite"
	@echo "  precommit  - run all pre-commit hooks"
	@echo "  up         - start the core stack (postgres, minio, mlflow, api)"
	@echo "  up-full    - start the full stack (adds airflow)"
	@echo "  down       - stop the whole stack"
	@echo "  bootstrap  - train the first model and set it as champion"
	@echo "  status     - registry aliases + versions"
	@echo "  promote    - move champion alias to the challenger"
	@echo "  rollback   - restore the previous champion"
	@echo "  logs       - tail stack logs"
	@echo "  data       - validate dataset in data/raw/ (manual download: README)"
	@echo "  train      - local training run (host, sqlite tracking)"
	@echo "  replay     - replay the stream (DAY_SECONDS=..., MAX=...)"
	@echo "  drift-*    - inject/inspect drift (drift-amount FACTOR=3 FROM_DAY=165)"
	@echo "               drift-nullify COLS=card4 FROM_DAY=165, drift-list, drift-off ID=1"
	@echo "  report-evidently - HTML drift report for the last window"
	@echo "  loadtest   - Locust load test of /score               [Phase 5]"
	@echo "  report     - regenerate all figures and tables        [Phase 5]"

install: ## Create/refresh the locked virtualenv
	uv sync

lint: ## Ruff lint + format check
	$(PY) ruff check .
	$(PY) ruff format --check .

typecheck: ## mypy over the package (config in pyproject.toml)
	$(PY) mypy

format: ## Apply ruff autofixes and formatting
	$(PY) ruff check --fix .
	$(PY) ruff format .

test: ## Run the test suite
	$(PY) pytest

precommit: ## Run all pre-commit hooks
	$(PY) pre-commit run --all-files

up: ## Start the core stack (postgres, minio, mlflow, api)
	$(COMPOSE) $(CORE) up -d --build
	@echo "api: http://localhost:8000  mlflow: http://localhost:5000  minio: http://localhost:9001"

up-full: ## Start the full stack (adds airflow on :8080)
	$(COMPOSE) $(FULL) up -d --build
	@echo "airflow: http://localhost:8080 (admin password: docker compose exec airflow cat ~/standalone_admin_password.txt)"

down: ## Stop the whole stack
	$(COMPOSE) $(CORE) $(FULL) down

logs: ## Tail stack logs
	$(COMPOSE) $(CORE) $(FULL) logs -f --tail 100

bootstrap: ## Train the first model and set it as champion
	$(COMPOSE) $(CORE) $(TOOLS) run --rm bootstrap train-champion --data-dir /data/raw

status: ## Registry aliases + versions
	$(COMPOSE) $(CORE) $(TOOLS) run --rm bootstrap status

promote: ## Move champion alias to the challenger
	$(COMPOSE) $(CORE) $(TOOLS) run --rm bootstrap promote

rollback: ## Restore the previous champion (TO=1 for an explicit version)
	$(COMPOSE) $(CORE) $(TOOLS) run --rm bootstrap rollback $(if $(TO),$(TO),)

data: ## Validate dataset presence and shape
	$(PY) python -m fraudops.data.check

train: ## Local training run (host, sqlite tracking)
	$(PY) python -m fraudops.models.train


replay: ## Replay the stream to Kafka (DAY_SECONDS=30 MAX= for caps)
	$(COMPOSE) $(FULL) $(TOOLS) run --rm replay \
	    --bootstrap-servers kafka:29092 --data-dir /data/raw \
	    --day-seconds $(or $(DAY_SECONDS),30) $(if $(MAX),--max-events $(MAX),)

drift-amount: ## Inject an amount shift (FACTOR=3 FROM_DAY=165)
	$(COMPOSE) $(FULL) $(TOOLS) run --rm --entrypoint python replay \
	    -m fraudops.monitoring.injector amount-factor \
	    --factor $(or $(FACTOR),3) --from-day $(or $(FROM_DAY),165)

drift-nullify: ## Null features from a sim day (COLS=card4 FROM_DAY=165)
	$(COMPOSE) $(FULL) $(TOOLS) run --rm --entrypoint python replay \
	    -m fraudops.monitoring.injector nullify \
	    --columns "$(or $(COLS),card4)" --from-day $(or $(FROM_DAY),165)

drift-list: ## List drift injections
	$(COMPOSE) $(FULL) $(TOOLS) run --rm --entrypoint python replay \
	    -m fraudops.monitoring.injector list

drift-off: ## Deactivate an injection (ID=1)
	$(COMPOSE) $(FULL) $(TOOLS) run --rm --entrypoint python replay \
	    -m fraudops.monitoring.injector deactivate --id $(ID)

report-evidently: ## HTML drift report for the last window
	$(COMPOSE) $(FULL) $(TOOLS) run --rm --entrypoint python replay \
	    -m fraudops.monitoring.evidently_report --data-dir /data/raw

report: ## Regenerate every figure and table from raw data [Phase 5]
	$(PY) python -m fraudops.reports.decay
	$(PY) python -m fraudops.reports.threshold_sensitivity
	$(PY) python -m fraudops.reports.detection_lag
	$(PY) python -m fraudops.reports.shap_report
	$(PY) python -m fraudops.reports.retrain_compare

loadtest: ## Locust load test of /score (USERS=50 RUN_TIME=2m) [Phase 5]
	$(PY) locust -f loadtest/locustfile.py --headless -u $(USERS) -r 5 -t $(RUN_TIME) \
	    --host http://localhost:8000 --csv reports/figures/locust --only-summary 2>&1 | tail -12
