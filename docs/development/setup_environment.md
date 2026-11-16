# Working Environmnet Setup

## Prerequisites

Before you proceed, make sure you have the following installed:

### Rust
We recommend using [rustup](https://rustup.rs/) for an easy setup. Minimum version: **1.76.0**

```bash
# For Linux/Mac:
curl --proto ‘=https’ --tlsv1.2 https://sh.rustup.rs -sSf | sh

# For Windows:
# Download and run the installer: https://rustup.rs/
```

### Python
Minimum version: **3.11**

```bash
# We recommend using uv for Python dependency management
# Install from: https://docs.astral.sh/uv/
```

> **Note:** `pymoors` uses [uv](https://docs.astral.sh/uv/) for dependency management. `moors` uses Cargo only.

## moors vs pymoors

```
moo-rs/
├── moors/    ← Rust crate
└── pymoors/  ← Python/pyo3 crate
```

**moors**
- A pure-Rust crate implementing multi-objective evolutionary algorithms and operators entirely in Rust.
- Mathematical core: defines fitness functions, sampling, crossover, mutation and duplicate cleaning as Rust structs and closures.
- High performance: leverages `ndarray`, `faer` and `rayon` for efficient numeric computation and parallelism.
- Rust-native API: consumed via `Cargo.toml` without FFI overhead.

**pymoors**
- A Python extension crate: uses [pyo3](https://github.com/PyO3/pyo3) to expose the complete `moors` core to Python.
- Pythonic interface: provides classes and functions that work seamlessly with NumPy arrays and the Python scientific ecosystem.
- Rapid prototyping: enables experimentation in notebooks or scripts while delegating compute-intensive work to Rust.
- Easy installation: `pip install pymoors` compiles and installs the bindings via Maturin.

---

## Working in `pymoors`

```sh
# dev build & install (uv + maturin)
make build-dev

# release build & install
make build-release

# format & lint Python (ruff)
make lint-python

# lint Rust bindings
make lint-rust

# run tests (excluding benchmarks)
make test

# run benchmark suite
make test-benchmarks

# build documentation site (mkdocs)
make docs
```

---

## Working in `moors`

```sh
# debug build
make build-dev

# optimized build
make build-release

# run tests
make test

# format code (cargo fmt)
make fmt

# check formatting / lint Rust
make lint

# run benchmarks
make bench
```

---

## From the Repository Root

### Monorepo Commands (Recommended)

The root `Makefile` provides convenient targets to work with the entire monorepo:

```sh
# Setup development environment (install dependencies & git hooks)
make setup

# Test everything (moors + pymoors)
make test

# Format code in both packages
make fmt

# Lint code in both packages
make lint

# Build in dev mode
make build-dev

# Build in release mode
make build-release

# Clean all artifacts
make clean
```

### Sub-package Specific Commands

If you need to run commands only for one package from the root:

```sh
# moors-specific (e.g., benchmarks)
make moors-bench
make moors-build-release

# pymoors-specific
make pymoors-build-dev
make pymoors-docs
```

Use `make help` at root to see all available targets.

---

## Contributing

```sh
# Fork & clone the repo
git clone https://github.com/your-username/moo-rs.git
cd moo-rs

# Create a feature branch
git checkout -b feat/your-feature-name

# Commit:
#   Use imperative messages, e.g. "feat: add new operator" or "fix: correct off-by-one"
#   Reference issues: "Fixes #123"

# Push & open PR:
git push origin feat/your-feature-name
```
