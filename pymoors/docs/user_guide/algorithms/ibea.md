# IBEA (Hypervolume-Indicator–Based Evolutionary Algorithm)

IBEA is a multi-objective evolutionary algorithm in which **selection pressure is driven entirely by a quality indicator** rather than Pareto ranking or density estimators. In the **hypervolume-based** variant (IBEA-H), fitness values derive from pairwise **hypervolume loss** contributions, which steers the search to **converge** toward the Pareto-optimal set while maintaining a **well-spread** approximation according to the hypervolume indicator.

Following Zitzler & Künzli (2004), fitness is assigned from pairwise indicator values and selection proceeds by iteratively removing the worst individual while updating the fitness of the remaining solutions. Using the hypervolume indicator provides a direct optimization signal aligned with many practical performance measures.

---

## Key Features

- **Indicator-Based Fitness Assignment:**
  Instead of nondominated sorting, IBEA computes a fitness value for each individual from **pairwise indicator contributions**. For IBEA-H, the indicator is derived from **hypervolume loss** if one solution is removed in favor of another. A common formulation is:

  $$
  F(x) = \sum_{y \neq x} -\exp\left( -\frac{I(y, x)}{\kappa} \right),
  $$

  where $I(y, x)$ is a **strictly monotone** indicator value (here, a function of hypervolume loss when replacing $x$ by $y$), and $\kappa > 0$ controls selection pressure.

- **Environmental Selection by Iterative Deletion:**
  IBEA forms a candidate set (typically the current population, or parent∪offspring if an elitist pool is used), then **repeatedly deletes the worst-metric individual**, updating impacted fitness values after each removal, until the target population size is reached. This procedure is **elitist** by construction, as higher-quality solutions persist.

- **Diversity via the Indicator:**
  Unlike crowding-distance methods, diversity emerges **implicitly** because hypervolume rewards sets that cover the Pareto front and expand the dominated volume. Regions that enlarge the dominated space receive higher preference.

- **Variation Operators:**
  Mating selection is typically tournament-based on fitness, while variation employs standard operators (e.g., **SBX** crossover and **polynomial mutation**) as in other evolutionary algorithms.

- **Computational Considerations:**
  The cost of fitness updates depends on the **hypervolume computation**. Exact hypervolume is efficient for **two objectives** ($\mathcal{O}(N \log N)$ with sorting) and becomes increasingly expensive with higher dimensions; practical implementations often use **Monte Carlo** or **FPR/FPO** estimators for $M \ge 4$ to trade accuracy for speed.

- **Constraint Handling:**
  In this implementation, constraints are handled using a **feasible-first** policy consistent with the constraint-domination principle:
  1. **Feasible vs. Infeasible:** Feasible solutions (all constraints satisfied) are always preferred.
  2. **Feasible–Feasible:** Compare by the IBEA fitness (hypervolume-based).
  3. **Infeasible–Infeasible:** Prefer the one with **smaller total violation**; ties broken by IBEA fitness computed on **penalized objectives**.

  Alternative strategies (e.g., $\varepsilon$-constraint or augmented Lagrangian penalties) can be substituted without altering the core selection mechanism.

---

## Implementation in `moo-rs`

In `moo-rs`, the IBEA implementation uses the **hypervolume indicator** as the core quality measure, with an **adaptive $\kappa$ parameter** to dynamically adjust selection pressure during evolution. This adaptation helps balance **exploration** and **exploitation** across generations.

### Normalization of Objectives
Before computing hypervolume contributions, all objective values are **normalized to [0,1]** using the **ideal** and **nadir points**:
- **Ideal point:** Best observed value for each objective.
- **Nadir point:** Worst observed value for each objective.

This normalization ensures that objectives are comparable and prevents bias toward any single dimension.


### IHD Indicator Based on Hypervolume
The **IHD-indicator** (Indicator based on Hypervolume Difference) measures the **impact of removing or replacing a solution** in terms of hypervolume contribution. It is defined as:

$$
IHD(x, y) =
\begin{cases}
IH(y) - IH(x), & \text{if } x \text{ dominates } y, \\
IH(\{x, y\}) - IH(x), & \text{otherwise}.
\end{cases}
$$

Where:
- **IH(S)** denotes the **hypervolume** of a set \( S \) with respect to a **reference point** \( r \).
- The hypervolume of a singleton $\{x\}$ is the **Lebesgue measure** of the region dominated by $\{x\}$ and bounded by $\{r \}$.

---

#### Interpretation of Terms

- **IH(x):**
  The hypervolume of the singleton set $\{x\}$. This is the volume of the hyperrectangle formed between the point $\{x\}$ and the reference point $\{r\}$.
  For a minimization problem with \( m \) objectives:
  $$
  IH(x) = \prod_{i=1}^{m} (r_i - f_i(x)),
  $$
  where $f_i(x)$ is the $i$-th objective value of solution $x$, and $r_i$ is the $i$-th component of the reference point.

- **IH(\{x, y\}):**
  The hypervolume of the set containing both $\{x\}$ and $\{y\}$. This accounts for the **union of dominated regions** by both points, avoiding double-counting overlapping areas.

---

#### Why Reference Point Matters

The reference point $r$ must be **worse than all solutions** in every objective dimension. If $r$ is too close to the Pareto front, extreme solutions will have **tiny hypervolume contributions**, which biases selection against them.
To avoid this, choose $r$ significantly larger than the normalized range (e.g., $[2, 2, ..., 2]$ after normalization to $[0,1]$).

---

#### Key Idea:
- If $x$ dominates $y$, the difference $IH(y) - IH(x)$ reflects the **loss in dominated space** when replacing $x$ with $y$.
- Otherwise, $IH(x, y) - IH(x)$ measures the **additional hypervolume** gained by adding $y$ to a set already containing $x$.

---

## EXPO2 Problem
**EXPO2** is a two-objective benchmark crafted to produce a **smooth, exponentially shaped Pareto front**, challenging algorithms to retain good coverage near the extremes where the trade-off is steep.

- **Decision Variables:**
  $\mathbf{x} = (x_1, \ldots, x_n)$, with $x_i \in [0, 1]$ for all $i$; a common setting is $n = 30$.

- **Objectives (minimization):**
  Let
  $$
  g(\mathbf{x}) = 1 + \frac{9}{n-1} \sum_{i=2}^{n} x_i.
  $$
  Then define
  $$
  f_1(\mathbf{x}) = x_1, \qquad
  f_2(\mathbf{x}) = g(\mathbf{x})\, \exp\!\left(-\frac{5\, x_1}{g(\mathbf{x})}\right).
  $$

- **Pareto-Optimal Front:**
  Achieved when $g(\mathbf{x}) = 1$ (i.e., $x_2 = \cdots = x_n = 0$), giving
  $$
  f_2 = \exp(-5 f_1), \quad f_1 \in [0, 1],
  $$
  which forms a **convex, exponentially decaying** front.

- **Key Characteristics:**
  - **Strongly Nonlinear Trade-Off:** The front is steep near $f_1 \approx 0$, stressing exploration of extreme solutions.
  - **Continuous and Convex:** No discontinuities; good for assessing **distribution** and **hypervolume growth**.
  - **Scalable Dimensionality:** As $n$ increases, **linkage** via $g(\mathbf{x})$ raises difficulty by coupling objectives with many variables.

---

## References

- Zitzler, E., & Künzli, S. (2004). *Indicator-Based Selection in Multiobjective Search*. In **PPSN VIII** (LNCS 3242, pp. 832–842). Springer.
- Zitzler, E., Thiele, L., Laumanns, M., Fonseca, C. M., & da Fonseca, V. G. (2003). *Performance Assessment of Multiobjective Optimizers: An Analysis and Review*. **TEO**/ETH Technical Report 139.
  (Introduces the hypervolume indicator as a robust performance measure.)
- While various IBEA variants exist (e.g., using the additive $\varepsilon$-indicator), this document focuses on the **hypervolume-based** instantiation commonly used in practice.
