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
