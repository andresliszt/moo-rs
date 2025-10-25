

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


## Key Features

- Implemented in Rust for superior performance.
- Accessible in Python through pyo3.
- Specialized in solving multi-objective optimization problems using genetic algorithms.


## Available Multi-Objective Algorithms

A concise index of the currently available algorithms.

| Algorithm | Description |
|---|---|
| [NSGA-II](../user_guide/algorithms/nsga2.md) | Baseline Pareto-based MOEA with fast non-dominated sorting and crowding distance. Robust, widely used for 2–3 objectives. |
| [NSGA-III](../user_guide/algorithms/nsga3.md) | Many-objective extension of NSGA-II using reference points to maintain diversity and guide convergence. |
| [IBEA](../user_guide/algorithms/ibea.md) | Indicator-Based EA that optimizes a quality indicator (e.g., hypervolume/ε-indicator) to drive selection. |
| [SPEA-II](../user_guide/algorithms/spea2.md) | Strength Pareto EA with enhanced fitness assignment, density estimation (k-NN), and external archive. |
| [AGEMOEA](../user_guide/algorithms/agemoea.md) | Approximation-guided MOEA that directly improves the Pareto-front approximation via set-level indicators. |
| [RNSGA-II](../user_guide/algorithms/rnsga2.md) | Reference-point oriented NSGA-II variant; biases the search toward regions of interest while preserving diversity. |
| [REVEA](../user_guide/algorithms/revea.md) | Reference vector/region–guided evolutionary algorithm using directional vectors to balance diversity and convergence. |
| [Custom Defined Algorithms](../user_guide/algorithms/custom/custom.md) | User defined algorithms by defining selection and survival operators|

## Introduction to Multi-Objective Optimization

**Multi-objective optimization** refers to a set of techniques and methods designed to solve problems where *multiple objectives* must be satisfied simultaneously. These objectives are often *conflicting*, meaning that improving one may deteriorate another. For instance, one might seek to **minimize** production costs while **maximizing** product quality at the same time.

### General Formulation

A multi-objective optimization problem can be formulated in a generic mathematical form. If we have \(k\) objective functions to optimize, it can be expressed as:

\[
\begin{aligned}
&\min_x \quad (f_1(x), f_2(x), \dots, f_k(x)) \\
&\text{subject to:} \\
&g_i(x) \leq 0, \quad i = 1, \dots, m \\
&h_j(x) = 0, \quad j = 1, \dots, p
\end{aligned}
\]

Where:
- \( x \) represents the set of decision variables.
- \( f_i(x) \) are the objective functions.
- \( g_i(x) \leq 0 \) and \( h_j(x) = 0 \) represent the **constraints_fn** of the problem (e.g., resource limits, quality requirements, etc.).

Unlike single-objective optimization, here we seek to optimize *all* objectives simultaneously. However, in practice, there is no single “best” solution for *all* objectives. Instead, we look for a set of solutions known as the **Pareto front** or **Pareto set**.

## Quickstart

The well known [ZTD3](../user_guide/algorithms/nsga2.md#zdt3-problem) problem solved with the [NSGA-II](../user_guide/algorithms/nsga2.md) algorithm!

=== "Rust"
    {% include-markdown "user_guide/algorithms/python/nsga2.md" %}

=== "Python"
    {% include-markdown "user_guide/algorithms/rust/nsga2.md" %}

## Advantages for Multi-Objective Optimization

1. **Natural Handling of Multiple Objectives**: By operating on a population of solutions, GAs can maintain an approximation to the **Pareto front** during execution.
2. **Flexibility**: They can be easily adapted to different kinds of problems (discrete, continuous, constrained, etc.).
3. **Robustness**: They tend to perform well in the presence of *noise* or uncertainty in the problem, offering acceptable performance under less-than-ideal conditions.

## Beauty and Misbehavior Optimization Problem

In this unique optimization problem, there is only one individual who optimizes both beauty and misbehavior at the same time: my little dog Arya!

<div style="text-align: center;">
  <img src="../images/arya.png" alt="Arya" width="500" />
</div>

Arya not only captivates with her beauty, but she also misbehaves in the most adorable way possible. This problem serves as a reminder that sometimes the optimal solution is as heartwarming as it is delightfully mischievous.
