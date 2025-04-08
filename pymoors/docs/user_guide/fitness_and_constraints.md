# Fitness Function in pymoors

In **pymoors**, the way to define objective functions for optimization is through a NumPy-based function that operates on an entire population. This means that the provided function, `f(genes)`, expects `genes` to be a 2D NumPy array with dimensions `(population_size, n_vars)`. It must then return a 2D NumPy array of shape `(population_size, n_objectives)`, where each row corresponds to the evaluation of a single individual.

This population-level evaluation is very important—it allows the algorithm to efficiently process and compare many individuals at once. When writing your fitness function, make sure it is vectorized and returns one row per individual, where each row contains the evaluated objective values.

Below is an example fitness function:

```python
import numpy as np
from pymoors.typing import TwoDArray

def fitness(genes: TwoDArray) -> TwoDArray:
    # Extract decision variables for each individual
    x1 = genes[:, 0]
    x2 = genes[:, 1]

    # Objective 1: Distance to (0,0)
    f1 = x1**2 + x2**2

    # Objective 2: Distance to (1,0)
    f2 = (x1 - 1)**2 + x2**2

    # Combine the two objectives into a single array,
    # where each row is the evaluation for one individual
    return np.column_stack([f1, f2])
```

Note that we have an special type alias `pymoors.typing.TwoDArray` which is no other thing than `Annotated[npt.NDArray[DType], "ndim=2"]`. This alias emphasizes that the array must be two-dimensional.

# Minimization and Maximization

In pymoors—as with many optimization frameworks—the core approach is based on **minimization**. This means that the optimization algorithm is designed to search for solutions that yield the lowest possible objective function values.

When you encounter a problem where you want to maximize an objective (for example, maximize profit or efficiency), you can simply convert it into a minimization problem. This is achieved by taking the negative of the objective function. In effect, maximizing an objective is equivalent to minimizing its negative value.

### Key Points

- **Default Minimization:**
  pymoors inherently minimizes objective functions. Lower values are considered better, and the optimization process works to reduce these values.

- **Converting Maximization to Minimization:**
  To handle maximization, you multiply the objective function by -1. By doing so, you transform a maximization problem into a minimization one. This unified approach simplifies the optimization framework.

- **Practical Considerations:**
  When defining your fitness functions, ensure that the returned evaluations conform to this minimization paradigm. For objectives that are originally maximization problems, adjust them by negating their outputs before they are returned by the fitness function.

This method ensures a consistent and streamlined optimization process within `pymoors`.

# Constraints

Constraints in an optimization problem are optional. They are defined using a similar approach to the fitness function. In pymoors, you define a constraint function `g(genes)` where `genes` is a 2D array of shape `(population_size, n_vars)`, and the function must return a 2D array of shape `(population_size, n_constraints)`. Each row of the output corresponds to the constraint evaluations for an individual in the population.

!!! warning "Feasibility of an Individual"

    In **pymoors**, an individual in the population is considered **feasible** if and only if all constraints are less than or equal to 0.

    In the following subsections, we will explain how to consider other types of inequalities.


Below is an example constraint function. In this example, we enforce a simple constraint: the sum of the decision variables for each individual must be less than or equal to a specified threshold value.

```python
import numpy as np
from pymoors.typing import TwoDArray

def constraints(genes: TwoDArray) -> TwoDArray:
    # Define a threshold for the sum of genes
    threshold = 10

    # Compute the sum for each individual (row)
    row_sums = np.sum(genes, axis=1)

    # Constraint: row_sums should be less than or equal to threshold.
    # We compute the violation as (row_sums - threshold). A value <= 0 means the constraint is satisfied.
    violations = row_sums - threshold

    # Return as a 2D array with one constraint per individual
    return violations.reshape(-1, 1)
```

### Key points

- **Input:**
The function receives `genes`, a 2D array with shape `(population_size, n_vars)`, where each row represents an individual in the population.

- **Constraint Calculation:**
For each individual, the function calculates the sum of its decision variables. The constraint is defined such that the sum must be less than or equal to a specified threshold (10 in this example).
- If the sum is less than or equal to the threshold, the constraint value (i.e., the violation) is ≤ 0, meaning the constraint is satisfied.
- If the sum exceeds the threshold, the resulting positive value indicates the magnitude of the violation.

- **Output:**
The function returns a 2D array of shape `(population_size, 1)`, where each element represents the constraint evaluation for the corresponding individual.

- **Note:**
The use of `reshape(-1, 1)` is crucial for ensuring that the output always has the correct dimensions, even when there is only one constraint. This guarantees consistency in the dimensionality of the constraint evaluations.


## Handling Constraints: Greater than 0 and Equality Constraints

In some optimization problems, constraints might be defined as either inequalities of the form `g(genes) > 0` or as equality constraints `g(genes) = 0`. In pymoors, since feasibility is determined by having all constraint values $≤ 0$, we handle these cases as follows:

- **Inequalities (`g(genes) > 0`):**
  This case is trivial to convert. Simply multiply the output of your constraint function by -1 to transform it into the standard form (`g(genes) <= 0`).

- **Equality Constraints (`g(genes) = 0`) with Tolerance:**
  Equality constraints are typically managed by allowing a small tolerance, `epsilon`, around 0. A common approach is to construct a penalty function by computing the squared deviation `(g(genes) - epsilon)²`. This squared term penalizes any deviation from the desired equality, while the tolerance `epsilon` provides some leeway for numerical imprecision.

### Example: Equality Constraint with Epsilon Tolerance

Below is an example constraint function. In this example, we enforce that the sum of decision variables for each individual should equal a specified threshold (10) within a tolerance of `epsilon`. The function computes the squared error from the target (adjusted by `epsilon`) and ensures the output is a 2D array using `reshape(-1, 1)`.

```python
import numpy as np
from pymoors.typing import TwoDArray

TOL = 1e-3

def constraints_equality(genes: TwoDArray) -> TwoDArray:
    threshold = 10
    row_sums = np.sum(genes, axis=1)
    # Compute the squared error from the equality constraint with tolerance epsilon.
    error = (row_sums - threshold - TOL)**2
    # The reshape ensures the output is a 2D array of shape (population_size, 1)
    return error.reshape(-1, 1)
```
