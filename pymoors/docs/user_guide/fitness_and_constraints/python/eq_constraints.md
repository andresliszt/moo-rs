```python

import numpy as np

from pymoors.typing import TwoDArray

EPSILON = 1e-6

def constraints(genes: TwoDArray) -> TwoDArray:
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

from pymoors import Constraints
from pymoors.typing import TwoDArray, OneDArray

EPSILON = 1e-6

def eq_constr(genes: TwoDArray) -> OneDArray:
    return genes[:, 0] + genes[:, 1] - 1

def ineq_constr(genes: TwoDArray) -> OneDArray:
    return genes[:, 0]**2 + genes[:, 1]**2 - 1

constraints = Constraints(eq = [eq_constr], ineq = [ineq_constr])

```
