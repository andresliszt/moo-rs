Also this class has two optional arguments `lower_bound` and `upper_bound` that will make each gene bounded by those values.

```python

import numpy as np

from pymoors import Constraints

EPSILON = 1e-6

def eq_constr(genes: np.ndarray):
    return genes[:, 0] + genes[:, 1] - 1

constraints = Constraints(eq = [eq_constr], lower_bound = 0.0, upper_bound = 1.0)

```
