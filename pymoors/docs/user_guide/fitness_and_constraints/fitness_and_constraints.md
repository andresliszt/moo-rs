# Fitness and Constraints

In multi-objective optimization, fitness functions assign a numerical score to each candidate solution based on how well it achieves the objectives, while constraint functions measure any violations of problem constraints (e.g., inequalities or equalities).

In `moors`/`pymoors`, both fitness and constraint functions are implemented as vectorized operations on ndarrays at the population level. That is, instead of evaluating one individual at a time, they accept a 2D array of shape (`num_individuals`, `num_vars`) and so that all individuals in the current population are evaluated simultaneously.

## Fitness

Fitness is a numerical measure of how well a candidate solution meets the optimization objectives.
It assigns each individual a score that guides selection and reproduction in evolutionary algorithms.

=== "Rust"

    {% include-markdown "user_guide/fitness_and_constraints/rust/fitness.md" %}

=== "Python"

    {% include-markdown "user_guide/fitness_and_constraints/python/fitness.md" %}

## Constraints

Feasibility is the key concept in constraints. This is very important in optimization, an individual is called *feasible* if and only if it satisfies all the constraints the problem defines. In `moors`/`pymoors` as in many other optimization frameworks, constraints allowed are evaluated as `<= 0.0`. In genetic algorithms, there are different ways to incorporate feasibility in the search for optimal solutions. In this framework, the guiding philosophy is: *feasibility dominates everything*, meaning that a feasible individual is always preferred over an infeasible one.

### Inequality Constraints

In `moors`/`pymoors` as mentioned, any output from a constraint function is evaluated as less than or equal to zero. If this condition is met, the individual is considered feasible. For constraints that are naturally expressed as greater than zero, the user should modify the function by multiplying it by -1, as shown in the following example

=== "Rust"
    {% include-markdown "user_guide/fitness_and_constraints/rust/ineq_constraints.md" %}

=== "Python"
    {% include-markdown "user_guide/fitness_and_constraints/python/ineq_constraints.md" %}

### Equality Constraints

As is many other frameworks, the known epsilon technique must be used to force $g(x) = 0$, select a tolerance $\epsilon$ and then transform $g$ into an inquality constraint

$$g_{\text{ineq}}(x) = \bigl|g(x)\bigr| - \varepsilon \;\le\; 0.$$

An example is given below

=== "Rust"
    {% include-markdown "user_guide/fitness_and_constraints/rust/eq_constraints.md" %}

=== "Python"
    {% include-markdown "user_guide/fitness_and_constraints/python/eq_constraints.md" %}

### Lower and Upper Bounds

=== "Rust"
    {% include-markdown "user_guide/fitness_and_constraints/rust/lower_upper_bounds.md" %}

=== "Python"
    {% include-markdown "user_guide/fitness_and_constraints/python/lower_upper_bounds.md" %}
