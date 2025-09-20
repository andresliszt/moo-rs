# Example: The Multi-Objective Knapsack Problem

The **multi-objective knapsack problem** is a classic example in optimization where we aim to select items, each with its own *benefits* and *costs*, subject to certain constraints_fn (e.g., weight capacity). In the multi-objective version, we want to optimize more than one objective function simultaneously—often, maximizing multiple benefits or qualities at once.

## Mathematical Formulation

Suppose we have $n$ items. Each item $i$ has:
- A profit $p_i$.
- A quality $q_i$.
- A weight $w_i$.

Let $(x_i)$ be a binary decision variable where $x_i = 1$ if item $i$ is selected and $x_i = 0$ otherwise. We define a knapsack capacity $C$. A common **multi-objective** formulation for this problem is:

$$
\begin{aligned}
&\text{Maximize} && f_1(x) = \sum_{i=1}^{n} p_i x_i \\
&\text{Maximize} && f_2(x) = \sum_{i=1}^{n} q_i x_i \\
&\text{subject to} && \sum_{i=1}^{n} w_i x_i \leq C,\\
& && x_i \in \{0, 1\}, \quad i = 1,\dots,n.
\end{aligned}
$$

=== "Rust"
    {% include-markdown "getting_started/rust/knapsack.md" %}

=== "Python"
    {% include-markdown "getting_started/python/knapsack.md" %}

!!! info
    Note that although the specified `population_size` was 16, the final population ended up
    being 13 individuals, of which 1 had `rank = 0`.
    This is because we used the `keep_infeasible` argument was set in false, removing any individual
    that did not satisfy the constraints_fn (in this case, the weight constraint).
    We also used a duplicate remover called `ExactDuplicatesCleaner` that eliminates all exact
    duplicates—meaning whenever `genes1 == genes2` in every component.

    **💡 Tip – Variable Types**

    In **pymoors**, there is no strict enforcement of whether variables are integer, binary, or
    real. The core Rust implementation works with `f64` ndarrays.
    To preserve a specific variable type—binary, integer, or real—you must ensure that the
    genetic operators themselves maintain it.

    It is **the user's responsibility** to choose the appropriate genetic operators for the
    variable type in question. In the knapsack example, we use binary-style genetic operators,
    which is why the solutions are arrays of 0 s and 1 s.
