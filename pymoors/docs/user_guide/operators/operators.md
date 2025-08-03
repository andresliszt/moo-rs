# Genetic Operators

Genetic operators are the core of any genetic algorithm; they are responsible for defining how individuals evolve across generations. They are formal mathematical operators defined on populations of candidate solutions, used to generate new individuals by recombining or perturbing existing ones.

## Mutation

Mutation is a unary genetic operator that introduces random perturbations into an individual’s representation. By randomly flipping, tweaking or replacing genes, mutation maintains population diversity and enables exploration of new regions in the search space.

=== "Rust"
    {% include-markdown "user_guide/operators/rust/mutation.md" %}

=== "Python"
    {% include-markdown "user_guide/operators/python/mutation.md" %}

## Crossover

Crossover is a binary genetic operator that combines genetic material from two parent individuals by exchanging segments of their representations, producing offspring that inherit traits from both parents. It promotes the exploration of new solution combinations while preserving useful building blocks.

=== "Rust"
    {% include-markdown "user_guide/operators/rust/crossover.md" %}

=== "Python"
    {% include-markdown "user_guide/operators/python/crossover.md" %}

## Sampling

Sampling is a genetic operator that generates new individuals by drawing samples from a defined distribution or the existing population.

=== "Rust"
    {% include-markdown "user_guide/operators/rust/sampling.md" %}

=== "Python"
    {% include-markdown "user_guide/operators/python/sampling.md" %}

## Selection

Selection is a genetic operator that chooses individuals from the current population based on their fitness, favoring higher-quality solutions for reproduction.

=== "Rust"
    {% include-markdown "user_guide/operators/rust/selection.md" %}

=== "Python"
    {% include-markdown "user_guide/operators/python/selection.md" %}

## Survival

Survival is a genetic operator that determines which individuals are carried over to the next generation based on a general quality criterion.

=== "Rust"
    {% include-markdown "user_guide/operators/rust/survival.md" %}

=== "Python"
    {% include-markdown "user_guide/operators/python/survival.md" %}
