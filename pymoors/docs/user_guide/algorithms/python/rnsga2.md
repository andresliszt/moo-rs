


















```python
import numpy as np
import matplotlib.pyplot as plt

from pymoors import (
    Rnsga2,
    RandomSamplingFloat,
    GaussianMutation,
    ExponentialCrossover,
    CloseDuplicatesCleaner,
    Constraints
)
from pymoors.schemas import Population
from pymoors.typing import TwoDArray

np.seterr(invalid="ignore")


def evaluate_ztd1(x: TwoDArray) -> TwoDArray:
    """
    Evaluate the ZTD1 objectives in a fully vectorized manner.
    """
    f1 = x[:, 0]
    g = 1 + 9.0 / (30 - 1) * np.sum(x[:, 1:], axis=1)
    f2 = g * (1 - np.power((f1 / g), 0.5))
    return np.column_stack((f1, f2))


def ztd1_theoretical_front():
    """
    Compute the theoretical Pareto front for ZTD1.
    """
    f1_theo = np.linspace(0, 1, 200)
    f2_theo = 1 - np.sqrt(f1_theo)
    return f1_theo, f2_theo


# Define two reference points (for example, points on the Pareto front)
reference_points = np.array([[0.5, 0.2], [0.1, 0.6]])

# Set up RNSGA-II algorithm with epsilon = 0.01
algorithm = Rnsga2(
    sampler=RandomSamplingFloat(min=0, max=1),
    crossover=ExponentialCrossover(exponential_crossover_rate=0.75),
    mutation=GaussianMutation(gene_mutation_rate=0.1, sigma=0.01),
    fitness_fn=evaluate_ztd1,
    constraints_fn=Constraints(lower_bound=0.0, upper_bound=1.0),
    duplicates_cleaner=CloseDuplicatesCleaner(epsilon=1e-8),
    num_vars=30,
    population_size=50,
    num_offsprings=50,
    num_iterations=700,
    mutation_rate=0.1,
    crossover_rate=0.9,
    keep_infeasible=False,
    reference_points=reference_points,
    verbose=False,
    epsilon=0.005,
    seed=1729,
)

# Run the algorithm
algorithm.run()

# Get the best Pareto front obtained (as a Population instance)
best: Population = algorithm.population.best_as_population
obtained_fitness = best.fitness

# Compute the theoretical Pareto front for ZTD1
f1_theo, f2_theo = ztd1_theoretical_front()

# Plot the theoretical Pareto front, obtained front, and reference points
plt.figure(figsize=(10, 6))
plt.plot(f1_theo, f2_theo, "k-", linewidth=2, label="Theoretical Pareto Front")
plt.scatter(
    obtained_fitness[:, 0],
    obtained_fitness[:, 1],
    c="r",
    marker="o",
    label="Obtained Front",
)
plt.scatter(
    [pt[0] for pt in reference_points],
    [pt[1] for pt in reference_points],
    marker="*",
    s=200,
    color="magenta",
    label="Reference Points",
)
plt.xlabel("$f_1$", fontsize=14)
plt.ylabel("$f_2$", fontsize=14)
plt.title("ZTD1 Pareto Front: Theoretical vs Obtained (RNSGA-II)", fontsize=16)
plt.legend()
plt.grid(True)
plt.show()
```


![png](../images/rnsga2_python.png)
