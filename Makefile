.DEFAULT_GOAL := help

#-------------------------------------------------
# Root Makefile for moors monorepo
#-------------------------------------------------

.PHONY: help pymoors-% moors-% \
	setup install \
	test test-moors test-pymoors \
	lint fmt lint-moors fmt-moors lint-pymoors fmt-pymoors fmt-rust-pymoors pyright \
	build build-dev build-release \
	build-moors-dev build-moors-release \
	build-pymoors-dev build-pymoors-release \
	docs docs-serve \
	clean \
	.check-tools .check-cargo .check-uv

#-------------------------------------------------
# Setup & Installation
#-------------------------------------------------

setup: .check-tools
	@echo "[Setup] Syncing Rust dependencies..."
	@cd moors && cargo fetch
	@echo "[Setup] Syncing Python dependencies..."
	@cd pymoors && uv sync --all-groups
	@$(MAKE) -C pymoors pre-commit-install
	@echo "✅ Setup complete!"

install: setup

.check-tools: .check-cargo .check-uv

.check-cargo:
	@cargo --version > /dev/null || \
		(echo "❌ Rust not installed. Install from: https://rustup.rs/" && exit 1)

.check-uv:
	@uv --version > /dev/null || \
		(echo "❌ uv not installed. Install from: https://docs.astral.sh/uv/" && exit 1)


test: test-moors test-pymoors
	@echo "✅ All tests passed!"

test-moors:
	@echo "[Testing] Running moors tests..."
	@$(MAKE) -C moors test

test-pymoors:
	@echo "[Testing] Running pymoors tests..."
	@$(MAKE) -C pymoors test

lint: lint-moors lint-pymoors

fmt: fmt-moors fmt-pymoors

lint-moors:
	@$(MAKE) -C moors lint

fmt-moors:
	@$(MAKE) -C moors fmt

lint-pymoors:
	@$(MAKE) -C pymoors lint-python
	@$(MAKE) -C pymoors lint-rust

fmt-pymoors:
	@$(MAKE) -C pymoors fmt-python

fmt-rust-pymoors:
	@$(MAKE) -C pymoors fmt-rust

pyright:
	@$(MAKE) -C pymoors pyright


build: build-dev

build-dev: build-moors-dev build-pymoors-dev

build-release: build-moors-release build-pymoors-release

build-moors-dev:
	@$(MAKE) -C moors build-dev

build-moors-release:
	@$(MAKE) -C moors build-release

build-pymoors-dev:
	@$(MAKE) -C pymoors build-dev

build-pymoors-release:
	@$(MAKE) -C pymoors build-release


docs:  ## Build documentation
	@echo "[docs] Building..."
	@uv sync --project docs
	@uv run --project docs mkdocs build --strict

docs-serve:  ## Serve documentation locally
	@echo "[docs] Serving at http://127.0.0.1:8000"
	@uv sync --project docs
	@uv run --project docs mkdocs serve

clean:  ## Clean all build artifacts
	@echo "Cleaning all artifacts..."
	@$(MAKE) -C moors clean
	@$(MAKE) -C pymoors clean



#-------------------------------------------------
# Help
#-------------------------------------------------

help:
	@echo "moors monorepo - Development Makefile"
	@echo
	@echo "Use 'make <target>' to run one of these targets:"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| grep -v '^\.' \
		| sed 's/:.*##/:/' \
		| column -t -s ':' \
		| sort
	@echo
	@echo "See pymoors/Makefile and moors/Makefile for available sub-targets."
