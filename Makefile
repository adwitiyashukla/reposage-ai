# RepoSage developer tasks. `make help` lists everything.
.DEFAULT_GOAL := help
.PHONY: help install dev test test-cov lint fmt typecheck check run index ask eval docker docker-run clean

PY ?= python
VENV ?= .venv
BIN := $(VENV)/bin
ifeq ($(OS),Windows_NT)
	BIN := $(VENV)/Scripts
endif

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create a virtualenv and install the package with all extras
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/pip install -e ".[dev,treesitter]"
	@echo "\nDone. Copy .env.example to .env and add your GEMINI_API_KEY."

dev: install ## Alias for install

test: ## Run the test suite (offline, no API key needed)
	$(BIN)/pytest -q

test-cov: ## Run tests with a coverage report
	$(BIN)/pytest --cov=reposage --cov-report=term-missing --cov-report=html

lint: ## Lint with ruff
	$(BIN)/ruff check src tests evals

fmt: ## Auto-fix lint issues and format
	$(BIN)/ruff check --fix src tests evals
	$(BIN)/ruff format src tests evals

typecheck: ## Static type check
	$(BIN)/mypy src/reposage

check: lint typecheck test ## Everything CI runs

run: ## Start the web UI and API on :8000
	$(BIN)/reposage serve --reload

index: ## Index this repository (make index SRC=owner/repo to index another)
	$(BIN)/reposage index $(or $(SRC),.)

ask: ## Ask a question (make ask Q="how does retrieval work?" R=reposage)
	$(BIN)/reposage ask -r $(or $(R),reposage-ai) "$(Q)"

eval: ## Run the evaluation suite against an index
	$(BIN)/python -m evals.run_evals --repo $(or $(R),reposage-ai) --output evals/REPORT.md

docker: ## Build the container image
	docker build -t reposage:latest .

docker-run: ## Run the container on :8000
	docker run --rm -p 8000:8000 --env-file .env -v reposage-data:/data reposage:latest

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
