# FraudOps — one command per task.
# Targets marked [Phase N] are placeholders until that phase lands.

PY := uv run

.DEFAULT_GOAL := help
.PHONY: help install lint format test precommit up down data train replay drift loadtest report

help: ## List available targets
	@echo "FraudOps Makefile targets:"
	@echo "  install   - create/refresh the locked virtualenv (uv sync)"
	@echo "  lint      - ruff check + format check"
	@echo "  format    - apply ruff autofixes and formatting"
	@echo "  test      - run the pytest suite"
	@echo "  precommit - run all pre-commit hooks"
	@echo "  up / down - start / stop the Docker stack            [Phase 2]"
	@echo "  data      - validate dataset in data/raw/            [manual download: see README Dataset section]"
	@echo "  train     - train baseline + threshold + evaluation  [Phase 1]"
	@echo "  replay    - replay the transaction stream            [Phase 3]"
	@echo "  drift     - inject synthetic drift                   [Phase 3]"
	@echo "  loadtest  - Locust load test of /score               [Phase 5]"
	@echo "  report    - regenerate all figures and tables        [Phase 5]"

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

up: ## Start the Docker stack [Phase 2]
	@echo "up: not yet implemented (arrives in Phase 2)"

down: ## Stop the Docker stack [Phase 2]
	@echo "down: not yet implemented (arrives in Phase 2)"

data: ## Validate dataset presence and shape
	$(PY) python -m fraudops.data.check

train: ## Train baseline, pick cost-optimal threshold, evaluate
	$(PY) python -m fraudops.models.train

replay: ## Replay transactions to Kafka in TransactionDT order [Phase 3]
	@echo "replay: not yet implemented (arrives in Phase 3)"

drift: ## Inject synthetic drift into the stream [Phase 3]
	@echo "drift: not yet implemented (arrives in Phase 3)"

loadtest: ## Locust load test of the scoring API [Phase 5]
	@echo "loadtest: not yet implemented (arrives in Phase 5)"

report: ## Regenerate every figure and table from raw data [Phase 5]
	@echo "report: not yet implemented (arrives in Phase 5)"
