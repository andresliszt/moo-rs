## Fitness

In **pymoors**, the way to define objective functions for optimization is through a [numpy](https://numpy.org/doc/). Currently, the only dtype supported is `float`, we're planning to relax this in the future. It means that when working with a different dtype, such as binary, its values must be trated as `float` (in this case as `0.0` and `1.0`).

This population-level evaluation is very important—it allows the algorithm to efficiently process and compare many individuals at once. When writing your fitness function, make sure it is vectorized and returns one row per individual, where each row contains the evaluated objective values.

Below is an example fitness function:

```python
import numpy as np

from pymoors.typing import TwoDArray

def fitness_dtlz2_3obj(genes: TwoDArray) -> TwoDArray:
    """
    DTLZ2 for 3 objectives (m = 3) with k = 0 (so num_vars = m−1 = 2):
    f1 = cos(π/2 ⋅ x0) ⋅ cos(π/2 ⋅ x1)
    f2 = cos(π/2 ⋅ x0) ⋅ sin(π/2 ⋅ x1)
    f3 = sin(π/2 ⋅ x0)
    """
    half_pi = np.pi / 2.0
    x0 = genes[:, 0] * half_pi
    x1 = genes[:, 1] * half_pi

    c0 = np.cos(x0)
    s0 = np.sin(x0)
    c1 = np.cos(x1)
    s1 = np.sin(x1)

    f1 = c0 * c1
    f2 = c0 * s1
    f3 = s0

    return np.stack([f1, f2, f3], axis=1)

```

## Constraints

Constraints are also a `numpy` function that evaluates at the population level. We recommend using the `pymoors.Constraints` class, which provides an appropriate wrapper, we will show this in the subsequent examples.

### Feasibility

This concept is very important in optimization, an individual is called *feasible* if and only if it satisfies all the constraints the problem defines. In `pymoors` as in many other optimization frameworks, constraints allowed are evaluated as `<= 0.0`. In genetic algorithms, there are different ways to incorporate feasibility in the search for optimal solutions. In this framework, the guiding philosophy is: *feasibility dominates everything*, meaning that a feasible individual is always preferred over an infeasible one.


### Inequality constraints

In `pymoors` as mentioned, any output from a constraint function is evaluated as less than or equal to zero. If this condition is met, the individual is considered feasible. For constraints that are naturally expressed as greater than zero, the user should modify the function by multiplying it by -1, as shown in the following example

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

### Equality constraints

As is many other frameworks, the known epsilon technique must be used to force $g(x) = 0$, select a tolerance $\epsilon$ and then transform $g$ into an inquality constraint

$$g_{\text{ineq}}(x) = \bigl|g(x)\bigr| - \varepsilon \;\le\; 0.$$

An example is given below

```python

import numpy as np

EPSILON = 1e-6

def constraints(genes: np.ndarray) -> np.ndarray:
    """
    Compute two constraints for each row [x, y] in genes:
      - Column 0: |x + y - 1| - EPSILON ≤ 0  (equality with ε-tolerance)
      - Column 1: x² + y² - 1.0 ≤ 0         (unit circle inequality)
    Returns an array of shape (n, 2).
    """
    # genes is expected to be shape (n_individuals, 2)
    x = genes[:, 0]
    y = genes[:, 1]

    # Constraint 1: |x + y - 1| - EPSILON
    eq = np.abs(x + y - 1.0) - EPSILON

    # Constraint 2: x^2 + y^2 - 1
    ineq = x**2 + y**2 - 1.0

    return np.stack((eq, ineq), axis=1)

```

This example ilustrates 2 constraints where one of them is an equality constraint. The `pymoors.Constraints` class lets us skip having to manually implement the epsilon technique; we can simply do


```python

import numpy as np

from pymoors import Constraints

EPSILON = 1e-6

def eq_constr(genes: np.ndarray):
    return genes[:, 0] + genes[:, 1] - 1

def ineq_constr(genes: np.ndarray)
    return genes[:, 0]**2 + genes[:, 1]**2 - 1

constraints = Constraints(eq = [eq_constr], ineq = [ineq_constr])

```

### Lower and upper bounds

Also this class has two optional arguments `lower_bound` and `upper_bound` that will make each gene bounded by those values.

```python

import numpy as np

from pymoors import Constraints

EPSILON = 1e-6

def eq_constr(genes: np.ndarray):
    return genes[:, 0] + genes[:, 1] - 1

constraints = Constraints(eq = [eq_constr], lower_bound = 0.0, upper_bound = 1.0)

```
