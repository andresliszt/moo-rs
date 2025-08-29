
In `pymoors`, the algorithms are implemented as classes that are exposed on the Python side with a set of useful attributes. These attributes include the final population and the optimal or best set of individuals found during the optimization process.

For example, after running an algorithm like NSGA2, you can access:
- **Final Population:** The complete set of individuals from the last generation.
- **Optimum Set:** Typically, the best individuals (e.g., those with rank 0) that form the current approximation of the Pareto front.

This design abstracts away the complexities of the underlying Rust implementation and provides an intuitive, Pythonic interface for setting up, executing, and analyzing multi-objective optimization problems.

```python

import numpy as np
from pymoors import (
    Nsga2,
    RandomSamplingFloat,
    GaussianMutation,
    SimulatedBinaryCrossover,
    CloseDuplicatesCleaner,
    Constraints
)
from pymoors.typing import OneDArray, TwoDArray


# Define the fitness function
def fitness(population_genes: TwoDArray) -> TwoDArray:
    x1 = population_genes[:, 0]
    x2 = population_genes[:, 1]
    # Objective 1: f1(x1,x2) = x1^2 + x2^2
    f1 = x1**2 + x2**2
    # Objective 2: f2(x1,x2) = (x1-1)^2 + x2**2
    f2 = (x1 - 1) ** 2 + x2**2
    return np.column_stack([f1, f2])


# Define the constraints_fn function
def constraints_fn(population_genes: TwoDArray) -> OneDArray:
    x1 = population_genes[:, 0]
    x2 = population_genes[:, 1]
    # Constraint 1: x1 + x2 <= 1
    g1 = x1 + x2 - 1
    # Convert to 2D array
    return g1

# Wrap the constraints in the special class and pass lower/upper bounds
constraints = Constraints(ineq = [constraints_fn], lower_bound = 0.0, upper_bound = 1.0)


# Set up the NSGA2 algorithm with the above definitions
algorithm = Nsga2(
    sampler=RandomSamplingFloat(min=0, max=1),
    crossover=SimulatedBinaryCrossover(distribution_index=5),
    mutation=GaussianMutation(gene_mutation_rate=0.1, sigma=0.01),
    fitness=fitness,
    num_objectives=2,
    constraints_fn=constraints,
    num_constraints=1,
    duplicates_cleaner=CloseDuplicatesCleaner(epsilon=1e-8),
    num_vars=2,
    population_size=200,
    num_offsprings=200,
    num_iterations=200,
    mutation_rate=0.1,
    crossover_rate=0.9,
    keep_infeasible=False,
    lower_bound=0,
    verbose=False,
)

# Run the algorithm
algorithm.run()
```
