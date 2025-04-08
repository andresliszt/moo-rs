.DEFAULT_GOAL := help

#-------------------------------------------------
# Rust moors crate Makefile
#-------------------------------------------------

.PHONY: build-dev build-release test fmt lint bench clean help

# Build in debug mode
build-dev:  ## cargo build (debug)
	@echo "[moors] build (debug)"
	cargo build

# Build in release mode
build-release:  ## cargo build --release
	@echo "[moors] build (release)"
	cargo build --release

# Run Rust tests
test:  ## cargo test
	@echo "[moors] test"
	cargo test

# Format
fmt:  ## cargo fmt + clippy
	@echo "[moors] format"
	cargo fmt

# Format and lint with rustfmt & clippy
lint:  ## cargo fmt + clippy
	@echo "[moors] lint"
	cargo fmt -- --check
	# TODO: enable clippy to CI/CD
	# cargo clippy -- -D warnings

# Run benchmarks
bench:  ## cargo bench -- --nocapture
	@echo "[moors] bench"
	cargo bench -- --nocapture

# Clean build artifacts
clean:  ## cargo clean + remove lcov
	@echo "[moors] clean"
	cargo clean
	rm -f lcov.info

# Show help
help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| sed 's/:.*##/:/' \
		| column -t -s ':' \
		| sort
