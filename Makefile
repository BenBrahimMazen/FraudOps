# FraudOps — one command per task.
# Targets marked [Phase N] are placeholders until that phase lands.

PY := uv run
COMPOSE := docker compose
CORE := --profile core
FULL := --profile full
TOOLS := --profile tools

.DEFAULT_GOAL := help
.PHONY: help install lint format test precommit up down up-full bootstrap status \
        promote rollback logs data train replay drift loadtest report

help: ## List available targets
	@echo "FraudOps Makefile targets:"
	@echo "  install    - create/refresh the locked virtualenv (uv sync)"
	@echo "  lint       - ruff check + format check"
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
	@echo "  replay     - replay the transaction stream            [Phase 3]"
	@echo "  drift      - inject synthetic drift                   [Phase 3]"
	@echo "  loadtest   - Locust load test of /score               [Phase 5]"
	@echo "  report     - regenerate all figures and tables        [Phase 5]"

install: ## Create/refresh the locked virtualenv
	uv sync

lint: ## Ruff lint + format check
	$(PY) ruff check .
	$(PY) ruff format --check .

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

rollback: ## Restore the previous champion
	$(COMPOSE) $(CORE) $(TOOLS) run --rm bootstrap rollback

data: ## Validate dataset presence and shape
	$(PY) python -m fraudops.data.check

train: ## Local training run (host, sqlite tracking)
	$(PY) python -m fraudops.models.train

replay: ## Replay transactions to Kafka in TransactionDT order [Phase 3]
	@echo "replay: not yet implemented (arrives in Phase 3)"

drift: ## Inject synthetic drift into the stream [Phase 3]
	@echo "drift: not yet implemented (arrives in Phase 3)"

loadtest: ## Locust load test of the scoring API [Phase 5]
	@echo "loadtest: not yet implemented (arrives in Phase 5)"

report: ## Regenerate every figure and table from raw data [Phase 5]
	@echo "report: not yet implemented (arrives in Phase 5]"
