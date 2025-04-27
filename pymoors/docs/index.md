---
hide:
  - navigation
---

# 📚 Welcome to the moo-rs Documentation

> This is the official documentation for the **moo-rs** project.

## What is moo-rs?

**moo-rs** is a project for implementing genetic algorithms to solve multi-objective optimization problems in Rust, leveraging the language’s full performance. The acronym “moo” stands for Multi-Objective Optimization. The repository houses `moors`, the mathematical and software core where all genetic algorithms are defined, and `pymoors`, the Python wrapper exposing the `moors` core via [pyo3](https://github.com/PyO3/pyo3).

- **moors**: a pure-Rust crate for high-performance implementations of genetic algorithms
- **pymoors**: a Python extension crate (via pyo3) exposing `moors` algorithms with a Pythonic API

> The use of `moors` in Rust is completely independent of Python

## Key Features

- Implemented in Rust for superior performance.
- Accessible in Python through pyo3.
- Specialized in solving multi-objective optimization problems using genetic algorithms.

## Introduction to Multi-Objective Optimization

**Multi-objective optimization** refers to a set of techniques and methods designed to solve problems where *multiple objectives* must be satisfied simultaneously. These objectives are often *conflicting*, meaning that improving one may deteriorate another. For instance, one might seek to **minimize** production costs while **maximizing** product quality at the same time.

## General Formulation

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
- \( g_i(x) \leq 0 \) and \( h_j(x) = 0 \) represent the **constraints** of the problem (e.g., resource limits, quality requirements, etc.).

Unlike single-objective optimization, here we seek to optimize *all* objectives simultaneously. However, in practice, there is no single “best” solution for *all* objectives. Instead, we look for a set of solutions known as the **Pareto front** or **Pareto set**.

## The Role of Genetic Algorithms in Multi-Objective Optimization

**Genetic algorithms (GAs)** are search methods inspired by *biological evolution* techniques, such as selection, crossover, and mutation. Some of their key characteristics include:

- **Heuristic**: They do not guarantee finding the global optimum but can often approximate it efficiently for many types of problems.
- **Parallel Search**: They work with a **population** of potential solutions, which allows them to explore multiple regions of the search space simultaneously.
- **Lightweight**: They usually require less information about the problem (e.g., no need for gradients or derivatives), making them suitable for complex or difficult-to-model objective functions.

### Advantages for Multi-Objective Optimization

1. **Natural Handling of Multiple Objectives**: By operating on a population of solutions, GAs can maintain an approximation to the **Pareto front** during execution.
2. **Flexibility**: They can be easily adapted to different kinds of problems (discrete, continuous, constrained, etc.).
3. **Robustness**: They tend to perform well in the presence of *noise* or uncertainty in the problem, offering acceptable performance under less-than-ideal conditions.

## Beauty and Misbehavior Optimization Problem

In this unique optimization problem, there is only one individual who optimizes both beauty and misbehavior at the same time: my little dog Arya!

<div style="text-align: center;">
  <img src="images/arya.png" alt="Arya" width="500" />
</div>

Arya not only captivates with her beauty, but she also misbehaves in the most adorable way possible. This problem serves as a reminder that sometimes the optimal solution is as heartwarming as it is delightfully mischievous.
