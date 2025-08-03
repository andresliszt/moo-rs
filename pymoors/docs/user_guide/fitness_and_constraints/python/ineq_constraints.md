```python

import numpy as np

from pymoors import Constraints


def constraints_sphere_lower_than_zero(population: np.ndarray) -> np.ndarray:
    """
    For each individual (row) in the population, compute x^2 + y^2 + z^2 - 1.
    Constraint is satisfied when this value is ≤ 0.
    """
    # population shape: (n_individuals, n_dimensions)
    sum_sq = np.sum(population**2, axis=1)
    return sum_sq - 1.0

def constraints_sphere_greater_than_zero(population: np.ndarray) -> np.ndarray:
    """
    For each individual (row) in the population, compute 1 - (x^2 + y^2 + z^2).
    Constraint is satisfied when this value is ≥ 0.
    """
    sum_sq = np.sum(population**2, axis=1)
    return 1.0 - sum_sq


constraints = Constraints(ineq = [constraints_sphere_lower_than_zero, constraints_sphere_greater_than_zero])

```

!!! warning "constraints as plain numpy function"

    In **pymoors**, you can pass to the algorithm a callable, in this scenario you must ensure that the return array is always
    2D, even if you work with just one constraint.
