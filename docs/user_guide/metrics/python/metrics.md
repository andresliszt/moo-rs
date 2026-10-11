To use quality metrics in `pymoors` just import the function you need; they accept NumPy arrays of shape `(n_individuals, n_objectives)` (and a 1D `(n_objectives,)` reference point for hypervolume), assuming **minimization**.

```python
import numpy as np

from pymoors import (
    generational_distance,
    hypervolume,
    inverted_generational_distance,
    spacing,
    spread,
)

front = np.array([[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]])
reference_front = front.copy()
reference_point = np.array([5.0, 5.0])

hv = hypervolume(front, reference_point)
gd = generational_distance(front, reference_front)
igd = inverted_generational_distance(front, reference_front)
sp = spacing(front)
delta = spread(front, reference_front)  # only defined for 2 objectives
```
