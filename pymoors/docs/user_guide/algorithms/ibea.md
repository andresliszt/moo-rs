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
