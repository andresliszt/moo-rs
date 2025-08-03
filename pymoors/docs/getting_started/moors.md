# moo-rs

> _Evolution is a mystery_

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python Versions](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
[![PyPI version](https://img.shields.io/pypi/v/pymoors.svg)](https://pypi.org/project/pymoors/)
[![crates.io](https://img.shields.io/crates/v/moors.svg)](https://crates.io/crates/moors)
[![codecov](https://codecov.io/gh/andresliszt/moo-rs/graph/badge.svg?token=KC6EAVYGHX)](https://codecov.io/gh/andresliszt/moo-rs)
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/andresliszt/moo-rs)

## Overview

`moo-rs` is a project for solving multi-objective optimization problems with evolutionary algorithms, combining:

- **moors**: a pure-Rust crate for high-performance implementations of genetic algorithms
- **pymoors**: a Python extension crate (via [pyo3](https://github.com/PyO3/pyo3)) exposing `moors` algorithms with a Pythonic API

Inspired by the amazing Python project [pymoo](https://github.com/anyoptimization/pymoo), `moo-rs` delivers both the speed of Rust and the ease-of-use of Python.

## Quickstart

The knapsack problem is a classic combinatorial optimization challenge where you must select a subset of items, each with a given weight and value, to include in a fixed‐capacity knapsack so as to maximize total value.
It requires balancing the trade‐off between item weights and values under a strict weight constraint.

In this small example, the algorithm finds a single solution on the Pareto front: selecting the items **(A, D, E)**, with a profit of **7** and a quality of **15**. This means there is no other combination that can match or exceed *both* objectives without exceeding the knapsack capacity (7).


=== "Rust"
    {% include-markdown "getting_started/rust/quick_start.md" %}

=== "Python"
    {% include-markdown "getting_started/python/quick_start.md" %}
