# Example: A Real-Valued Multi-Objective Optimization Problem

Below is a simple *two-variable* multi-objective problem to illustrate real-valued optimization with `pymoors`. We have two continuous decision variables, $x_1$ and $x_2$, both within a given range. We define **two** objective functions to be minimized simultaneously, and we solve this using the popular NSGA2 algorithm.

## Mathematical Formulation

Let $\mathbf{x} = (x_1, x_2)$ be our decision variables, each constrained to the interval $[-2, 2]$. We define the following objectives:

$$
\begin{aligned}
\min_{\mathbf{x}} \quad
&f_1(x_1, x_2) = x_1^2 + x_2^2 \\
\min_{\mathbf{x}} \quad
&f_2(x_1, x_2) = (x_1 - 1)^2 + x_2^2 \\
\text{subject to} \quad
& -2 \le x_1 \le 2, \\
& -2 \le x_2 \le 2.
\end{aligned}
$$

**Interpretation**

1. **$f_1$** measures the distance of $\mathbf{x}$ from the origin $(0,0)$ in the 2D plane.
2. **$f_2$** measures the distance of \(\mathbf{x}\) from the point $(1,0)$.

Thus, $\mathbf{x}$ must compromise between being close to $(0,0)$ and being close to $(1,0)$. There is no single point in $[-2,2]^2$ that *simultaneously* minimizes both distances perfectly (other than at the boundary of these trade-offs), so we end up with a **Pareto front** rather than a single best solution.


=== "Rust"
    {% include-markdown "getting_started/rust/real_valued.md" %}

=== "Python"
    {% include-markdown "getting_started/python/real_valued.md" %}
