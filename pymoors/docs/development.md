# Development Guide

## Prerequisites

Before you proceed, make sure you have **Rust** installed. We recommend using [rustup](https://rustup.rs/) for an easy setup:

```bash
# For Linux/Mac:
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh

# For Windows:
# Download and run the installer:
# https://rustup.rs/
```

Also `pymoors` uses [uv](https://github.com/astral-sh/uv). Make sure it’s available on your `PATH` so the `make` commands can run properly.

> **Note:** `moors` doesn’t use `uv`

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

## From the Repo Root

```sh
# run pymoors dev build from root
make pymoors-build-dev

# run pymoors lint-Python from root
make pymoors-lint-python

# run pymoors tests from root
make pymoors-test

# run moors tests from root
make moors-test

# run moors benchmarks from root
make moors-bench
```

Use `make help` at root to list all available `pymoors-<target>` and `moors-<target>` commands.

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
