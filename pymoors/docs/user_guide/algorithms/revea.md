# REVEA

REVEA (Reference Vector Guided Evolutionary Algorithm) is a many-objective evolutionary algorithm designed to efficiently approximate the Pareto front by dynamically adapting a set of reference vectors. By guiding the search with these vectors, REVEA maintains a balance between convergence toward optimality and diversity across the objective space.

## Key Features

- **Dynamic Reference Vectors:**
  REVEA starts with an initial set of reference vectors and periodically updates them using the current population’s extreme objective values (the ideal and nadir points). This update mechanism ensures that the reference vectors remain aligned with the evolving search space.

- **Angle Penalized Distance (APD):**
  A core component of REVEA is the Angle Penalized Distance metric, which combines the angular deviation (denoted by $\theta$, the angle between a solution and its associated reference vector) with a scaling factor that adapts over generations. In essence, APD favors solutions that are both close in direction to the reference vector and robust in magnitude.

- **Reference Vector Association:**
  Each solution is associated with the reference vector with which it has the highest cosine similarity. This association partitions the population into niches, promoting a uniform spread of solutions along the Pareto front.

- **Adaptive Selection:**
  In the survival selection phase, REVEA selects, from each niche, the solution with the smallest APD. This elitist approach helps ensure that all regions of the objective space are well represented in the next generation.

## Reference Vector Association and APD

### Association Mechanism

Let ${\mathbf{v}_1^t, \mathbf{v}_2^t, \dots, \mathbf{v}_N^t}$ be the reference vector set at generation $t$. For each solution $\mathbf{f}(\mathbf{x})$, the cosine similarity with each reference vector is computed. The solution is then assigned to the reference vector for which the angular difference (i.e. the angle $\theta$ between them) is minimized.

### Angle Penalized Distance (APD)

For a solution $\mathbf{f}_i$ associated with reference vector $\mathbf{v}_j^t$, the Angle Penalized Distance is given by:

$$
\text{APD}_{ij} = \Bigl( 1 + M \Bigl(\frac{t}{t_{\max}}\Bigr)^\alpha \cdot \frac{\theta_{ij}}{\gamma_j} \Bigr) \cdot \|\mathbf{f}_i\|
$$

where:
- $M$ is the number of objectives,
- $t$ is the current generation and $t_{\max}$ is the maximum number of generations,
- $\alpha$ is a control parameter,
- $\gamma_j$ is a scaling factor for the $j$-th reference vector (often computed as the minimum inner product between $\mathbf{v}_j^t$ and the other reference vectors),
- $\theta_{ij}$ is the angle between the solution $\mathbf{f}_i$ and the reference vector $\mathbf{v}_j^t$, and
- $\|\mathbf{f}_i\|$ is the norm of the (possibly translated) objective vector of the solution.

**Brief Interpretation of APD:**
The APD metric measures how well a solution aligns with its associated reference vector. A lower APD indicates that the solution is not only close in angle (small $\theta_{ij}$) to the reference direction, reflecting good convergence, but also has a desirable magnitude. The term $M\Bigl(\frac{t}{t_{\max}}\Bigr)^\alpha$ gradually increases the penalty on angular deviation over generations, thus promoting diversity in the early stages and fine convergence later on.

### Reference Vector Update

At a predefined frequency $fr$, the reference vectors are updated to better reflect the current objective landscape. Given the ideal point $z_{\min}$ and the nadir point $z_{\max}$ of the current population, the updated reference vector $\mathbf{v}_i^{t+1}$ is computed as:

$$
\mathbf{v}_i^{t+1} = \frac{\mathbf{v}_i^0 \circ (z_{\max} - z_{\min})}{\|\mathbf{v}_i^0 \circ (z_{\max} - z_{\min})\|}
$$

where:
- $\mathbf{v}_i^0$ is the initial reference vector,
- $\circ$ denotes the element-wise (Hadamard) product.

## Selection Process

- **Niche Formation:**
  Each solution is assigned to a niche based on its closest reference vector (i.e., the one with the smallest angle $\theta$).

- **Elitist Survival:**
  Within each niche, the solution with the smallest APD is selected to continue to the next generation, ensuring that all regions of the objective space contribute to the evolving Pareto front.

- **Dynamic Adaptation:**
  By periodically updating the reference vectors based on the current population’s extreme values, REVEA maintains effective guidance even as the search space shifts over time.

## Summary

REVEA leverages dynamic reference vector adaptation and the innovative Angle Penalized Distance metric to balance convergence and diversity in many-objective optimization. By continuously realigning its reference vectors with the evolving objective landscape and selecting solutions that are both close in angle (small $\theta$) and robust in performance, REVEA offers an effective strategy for tackling complex multi-objective problems.

=== "Rust"
    {% include-markdown "user_guide/algorithms/rust/revea.md" %}

=== "Python"
    {% include-markdown "user_guide/algorithms/python/revea.md" %}
