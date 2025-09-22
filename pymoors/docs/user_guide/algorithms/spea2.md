# SPEA-II

SPEA-II is an elitist multi‐objective evolutionary algorithm that, unlike NSGA-II, does **not** use Pareto fronts. Instead, at each generation it splits the combined parents + offspring set into **non-dominated** and **dominated** individuals and maintains a separate archive of elites.

## Key Features

- **Strength & Raw Fitness**
  1. **Strength** of individual $i$:
     $$S(i) = \bigl|\{\,j \in P \cup A \mid i \text{ dominates } j\}\bigr|$$
     where $P$ is the current population (size $N$) and $A$ the elite archive.
  2. **Raw fitness** of $i$:
     $$R(i) = \sum_{\substack{j \in P \cup A\\j \text{ dominates } i}} S(j)\,. $$

- **Density Estimation**
  For tie-breaking among equal $R(i)$, compute
  $$D(i) = \frac{1}{\sigma_i^k + 2},$$
  where $\sigma_i^k$ is the distance to the $k$-th nearest neighbor in objective space. Typically $k = \lfloor\sqrt{N + A_{\max}}\rfloor$.

- **Elitist Archive**
  - Fixed size $A_{\max}$ (often set equal to $N$).
  - Each generation:
    1. Form $Q = P \cup A$.
    2. Extract all **non-dominated** from $Q$ → provisional $A'$.
    3. If $|A'| > A_{\max}$, truncate by iteratively removing the individual with the smallest $D(i)$ until $|A'| = A_{\max}$.
    4. If $|A'| < A_{\max}$, fill up with the best **dominated** individuals from $Q\setminus A'$ in ascending order of $F(i)=R(i)+D(i)$.

- **Environmental Selection for Mating**
  - Parents are selected **only** from the updated archive $A_{t+1}$ (size $A_{\max}$) by binary‐tournament on $F(i)=R(i)+D(i)$.
  - Generate exactly $N$ offspring (so offspring count = population size).

- **Survival of Population**
  - **Population replacement** is generational: after mating, the new population $P_{t+1}$ consists **solely** of the $N$ offspring (no mixing with $P_t$).

- **Constraint Handling**
  For constrained problems:
  1. **Feasible** solutions always outrank infeasible ones in both archive update and tournament.
  2. In ties raw fitness $R(i)$ is heavily penalized.
  3. Density $D(i)$ is computed only within the feasible set to preserve feasible diversity.

---

## Implementation in moo-rs

In pymoors the algorithm does not maintain a separate set A — instead, P_t itself serves as the elite archive of best individuals. Each iteration proceeds as follows:

1. **Offspring generation**
   An offspring set $O_t$ of size O is produced by binary-tournament selection on $P_t$ using the composite fitness
   $F(i) = R(i) + D(i)$.

2. **Combine parents and offspring**
   The union $Q = P_t ∪ O_t$ is formed.

3. **Compute metrics**
   For every individual in $Q$, compute:
   - Strength $S(i)$
   - Raw fitness $R(i)$
   - Density $D(i)$

4. **Build new population/archive**
   Extract all non-dominated individuals from $Q$.
   - If $\text{|non-dominated|} > N$, remove those with smallest $D(i)$ one by one until exactly $N$ remain.
   - If $\text{|non-dominated|} < N$, fill up with the best dominated individuals sorted by increasing $F(i)$ until the total is $N$.

5. **Advance to next generation**
   The selected $N$ individuals become $P_{t+1}$, which also functions as the archive for the next iteration.

This design lets the population evolve generation by generation while concurrently acting as the elite memory, without any additional archive structure.


## ZDT6 Problem

**ZDT6** is a challenging two-objective benchmark that tests an algorithm’s ability to handle highly non-linear, multi-modal behavior and a non-uniform Pareto front.

- **Two Conflicting Objectives:**
  - $f_{1}(\mathbf{x}) = 1 - \exp\bigl(-4\,x_{1}\bigr)\,\bigl[\sin\bigl(6\pi\,x_{1}\bigr)\bigr]^{6}$
  - $f_{2}(\mathbf{x}) = g(\mathbf{x})\,h\bigl(f_{1}(\mathbf{x}),\,g(\mathbf{x})\bigr)$

- **Auxiliary Functions:**
  - $g(\mathbf{x}) = 1 + 9\,\dfrac{\sum_{i=2}^{n} x_{i}}{n - 1}$
  - $h\bigl(f_{1}, g\bigr) = 1 - \bigl(\tfrac{f_{1}}{g}\bigr)^{2}$

- **Key Characteristics:**
  - **Multi-modal front shape:** $f_{1}$ has many local optima due to the $\sin^{6}(6\pi\,x_{1})$ term, making convergence difficult.
  - **Non-uniform Pareto front:** Optimal solutions concentrate in a narrow region of the objective space, challenging diversity preservation.
  - **Biased distribution:** Pareto-optimal $x_{1}$ values are skewed toward the middle of the decision range.
  - **Non-convexity:** The $h$ function introduces strong curvature, requiring careful balance between convergence and spread.

- **Domain:**
  Each decision variable $x_{i}\in[0,1]$, typically with $n=30$.

ZDT6 is ideal for evaluating how well an algorithm balances exploration of multiple local optima in $f_{1}$ with exploitation toward a concentrated, non-uniform Pareto front.

=== "Rust"
    {% include-markdown "user_guide/algorithms/rust/spea2.md" %}

=== "Python"
    {% include-markdown "user_guide/algorithms/python/spea2.md" %}
