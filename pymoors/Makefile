.DEFAULT_GOAL := help
SOURCES      = python/pymoors tests

#-------------------------------------------------
# Environment checks & pre‑commit helpers
#-------------------------------------------------
.PHONY: .uv .pre-commit pre-commit-install pre-commit-run
.uv:     ## Check for uv
	@uv -V \
	  || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'
.pre-commit:   ## Check for pre-commit
	@uv run pre-commit --version \
	  || echo 'Please install pre-commit: https://pre-commit.com/'

pre-commit-install: .pre-commit   ## Install git hooks via pre-commit
	@echo "Installing pre-commit hooks…" \
	  && uv run pre-commit install

pre-commit-run:                   ## Run all enabled hooks against all files
	@echo "Running pre-commit on all files…" \
	  && uv run pre-commit run --all-files

#-------------------------------------------------
# Python/pyo3 (pymoors)
#-------------------------------------------------
.PHONY:  lock build-dev build-release \
        lint-python lint-rust pyright \
        test test-benchmarks test-all \
        coverage docs

lock:   ## Rebuild Python lockfile
	@echo "[pymoors] rebuilding lockfile" \
	  && uv lock --upgrade

build-dev:  ## Build & install in dev mode
	@echo "[pymoors] dev build" \
	  && uv sync --group dev \
	  && uv run maturin develop --uv

build-release:  ## Build & install in release mode
	@echo "[pymoors] release build" \
	  && uv sync --group release \
	  && uv run maturin develop --release --uv

lint-python:  ## Lint Python code
	@echo "[pymoors] lint python" \
	  && uv sync --group linting \
	  && uv run ruff check $(SOURCES) \
	  && uv run ruff format --check $(SOURCES)

lint-rust:  ## Lint Rust bindings
	@echo "[pymoors] lint rust" \
	  && uv run cargo fmt -- --check

pyright:  ## Run Pyright type checks
	@echo "[pymoors] typecheck" \
	  && uv sync --group linting \
	  && uv run pyright

test:  ## Run pytest excluding benchmarks
	@echo "[pymoors] test (no benchmarks)" \
	  && uv sync --group testing \
	  && uv run pytest --ignore=tests/benchmarks

test-benchmarks:  ## Run only benchmarks tests
	@echo "[pymoors] benchmarks only" \
	  && uv sync --group testing \
	  && uv run pytest tests/benchmarks --codspeed

test-all:  ## Run all tests including benchmarks
	@echo "[pymoors] all tests" \
	  && uv sync --group testing \
	  && uv run pytest

coverage:  ## Run coverage & generate lcov
	@echo "[pymoors] coverage" \
	  && uv sync --group testing \
	  && uv run coverage run -m pytest --ignore=tests/benchmarks \
	  && uv run coverage xml \
	  && coverage-prepare lcov python/pymoors/*.so

docs:  ## Build documentation
	@echo "[pymoors] docs" \
	  && uv sync --group docs \
	  && uv run mkdocs build --strict

#-------------------------------------------------
# Misc
#-------------------------------------------------
clean:  ## Remove all build artifacts
	@echo "Cleaning all artifacts…" \
	  && rm -f python/pymoors/*.so \
	  && find . -type d -name '__pycache__' -exec rm -rf {} + \
	  && find . -type f -name '*.py[co]' -delete \
	  && find . -type f -name '*~' -delete \
	  && find . -type f -name '.*~' -delete \
	  && rm -rf python/pymoors/*.egg-info .cache htmlcov .pytest_cache \
	  && rm -f .coverage .coverage.* \
	  && rm -rf build perf.data*

help:  ## Show this help
	@grep -E '^[A-Za-z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | sed 's/:.*##/:/' \
	  | column -t -s ':' \
	  | sort
