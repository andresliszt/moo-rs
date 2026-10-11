# Quality Metrics

A genetic algorithm for multi-objective optimization doesn't return a single "best" solution — it returns an approximation of the **Pareto front**: a set of mutually non-dominated trade-off solutions. Comparing two such sets (e.g. two algorithms, or two runs with different hyperparameters) by eye only works in 2 objectives. **Quality indicators** turn a front into one or more numbers so fronts can be ranked and compared objectively.

Broadly, indicators fall into three families:

- **Convergence**: how close is the obtained front to the true/optimal one?
- **Diversity / spread**: how uniformly and how widely do the points cover the front?
- **Combined**: a single number capturing both convergence and diversity (e.g. hypervolume).

`moors` implements five of the most widely used indicators, all under the convention that **every objective is minimized** and fitness matrices have shape `(n_individuals, n_objectives)`.

| Metric | Needs reference front? | Needs reference point? | Captures |
|--------|:---:|:---:|----------|
| [Hypervolume (HV)](#hypervolume-hv) | no | yes | convergence + diversity |
| [Generational Distance (GD)](#generational-distance-gd) | yes | no | convergence |
| [Inverted Generational Distance (IGD)](#inverted-generational-distance-igd) | yes | no | convergence + coverage |
| [Spacing (SP)](#spacing-sp) | no | no | diversity (uniformity) |
| [Spread (Δ)](#spread) | yes | no | diversity (uniformity + extent), 2 objectives only |

<div class="lang-rust" markdown>

{% include-markdown "user_guide/metrics/rust/metrics.md" %}

</div>

<div class="lang-python" markdown>

{% include-markdown "user_guide/metrics/python/metrics.md" %}

</div>

## Hypervolume (HV)

For a minimization front $A$ and a reference point $r$ dominated by every point of $A$, the hypervolume is the Lebesgue measure of the region jointly dominated by $A$ and bounded above by $r$:

$$
HV(A, r) = \Lambda\left( \bigcup_{a \in A} [a, r] \right)
$$

where $[a, r]$ is the axis-aligned hyper-rectangle with corners $a$ and $r$, and $\Lambda$ is the Lebesgue measure (area in 2D, volume in 3D, ...).

Hypervolume is the only indicator in common use that is **strictly Pareto-compliant**: if front $A$ weakly dominates front $B$, then $HV(A) \geq HV(B)$, with equality only when the fronts are equal. This makes it the de-facto standard for comparing MOO algorithms, at the cost of needing a reference point and being more expensive to compute as the number of objectives grows.

`moors` computes HV with the recursive dimension-slicing algorithm ("Hypervolume by Slicing Objectives", While, Hingston, Barone & Huband, 2006). Any point of $A$ that is worse than $r$ in some objective is clamped to $r$ in that objective (i.e. it simply contributes no extra volume there).

!!! info "Higher is better"
    Larger hypervolume means a front that converges further and/or spreads wider.

## Generational Distance (GD)

Introduced by Van Veldhuizen & Lamont (1998), GD measures how far an obtained front $A$ is from a reference front $Z$ (the true Pareto front, or a very dense approximation of it):

$$
GD(A, Z) = \frac{\sqrt{\sum_{a \in A} d(a, Z)^2}}{|A|}, \qquad d(a, Z) = \min_{z \in Z} \lVert a - z \rVert_2
$$

$GD = 0$ if and only if every point of $A$ lies exactly on $Z$. GD says nothing about *coverage*: a single point sitting exactly on $Z$ gives $GD=0$ regardless of how little of the front it represents — this is why GD is usually reported alongside a diversity metric, or IGD is used instead.

!!! info "Lower is better"

## Inverted Generational Distance (IGD)

Introduced by Coello Coello & Cortés (2005), IGD swaps the roles of GD, measuring the average distance from each reference point to the nearest solution actually obtained:

$$
IGD(A, Z) = \frac{\sum_{z \in Z} d(z, A)}{|Z|}, \qquad d(z, A) = \min_{a \in A} \lVert z - a \rVert_2
$$

Because *every* reference point must be close to some solution in $A$, a front with gaps or poor coverage of $Z$ is penalized even if the points it does have are perfectly converged — unlike GD. This makes IGD a good single proxy for both convergence and diversity when a reference front is available.

!!! info "Lower is better"

## Spacing (SP)

Schott's spacing (1995) measures how **uniformly** solutions are distributed along a front, without requiring any reference front:

$$
SP(A) = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n} (\bar d - d_i)^2}
$$

where $d_i = \min_{j \neq i} \sum_{k=1}^{M} \lvert f_k(i) - f_k(j) \rvert$ is the $L_1$ (Manhattan) distance from solution $i$ to its nearest neighbor in objective space, and $\bar d$ is the mean of all $d_i$.

$SP = 0$ means every nearest-neighbor distance is identical (a perfectly even spread); a larger $SP$ indicates clustering or uneven coverage. Because it needs no reference front, spacing is useful to monitor diversity during/after a single run when the true Pareto front is unknown.

!!! info "Lower is better"

## Spread (Δ)

Introduced in the original NSGA-II paper (Deb, Pratap, Agarwal & Meyarivan, 2002), spread combines **extent** and **uniformity** of a front $A$ against a reference front $Z$. Sorting $A$ by its first objective:

$$
\Delta(A, Z) = \frac{d_f + d_l + \sum_{i=1}^{n-1} \lvert d_i - \bar d \rvert}{d_f + d_l + (n-1)\bar d}
$$

where $d_i$ is the Euclidean distance between consecutive solutions $i$ and $i+1$, $\bar d$ is their mean, and $d_f$/$d_l$ are the Euclidean distances between the extreme (boundary) solutions of $A$ and the extreme solutions of $Z$.

$\Delta = 0$ for a perfectly uniform front that also reaches both extremes of $Z$; larger values indicate gaps, clustering, or a front that fails to reach the true extremes.

!!! warning "2 objectives only"
    "Consecutive solutions along the front" is only unambiguous when sorting by a single objective, so `moors` only implements Δ for 2-objective fronts.

!!! info "Lower is better"
