# Algorithms

Genetic algorithms are the core of the entire project; very briefly, an algorithm can be defined in the following steps

 **Initialize:** create a population `P` of `μ` random valid solutions.
- **Evaluate:** compute fitness for each individual.
- **Selection:** pick parents biased to higher fitness (e.g., tournament or rank).
- **Mating:** recombine parents to make offspring (crossover).
- **Mutation:** randomly perturb offspring (small, bounded changes).
- **Evaluate offspring:** compute their fitness and constraints (if given).
- **Survival:** build next generation of size `μ`
  - `(μ+λ)` elitist: keep best from parents ∪ offspring
  - `(μ,λ)` generational: keep best offspring only
- **Stop:** when max generations/time reached or no improvement.

!!! example "Genetic Algorithm"
    ```python
    # Pseudo code of a genetic algorithm
    P = init(μ); evaluate(P)
    repeat:
        parents = select(P)
        children = mutate(crossover(parents))
        evaluate(children)
        P = survive(P, children, μ)   # (μ+λ) or (μ,λ)
    until stop
    ```


## Mathematical Formulation of a Multi-Objective Optimization Problem with Constraints

Consider the following optimization problem

$$
\begin{aligned}
\min_{x_1, x_2} \quad & f_1(x_1,x_2) = x_1^2 + x_2^2 \\
\min_{x_1, x_2} \quad & f_2(x_1,x_2) = (x_1-1)^2 + x_2^2 \\
\text{subject to} \quad & x_1 + x_2 \leq 1, \\
& x_1 \geq 0,\quad x_2 \geq 0.
\end{aligned}
$$

=== "Rust"
    {% include-markdown "user_guide/algorithms/introduction/rust-introduction.md" %}

=== "Python"
    {% include-markdown "user_guide/algorithms/introduction/python-introduction.md" %}
