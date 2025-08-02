To use close duplicates cleaner in `pymoors` just pass to the algorithm the cleaner object

```python
import numpy as np

from pymoors import CloseDuplicatesCleaner

population = np.array([[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]])
reference = np.array([[1.01, 2.01, 3.01]]) # close to row 0 of population
epsilon = 0.05;

cleaner = CloseDuplicatesCleaner(epsilon=1e-5)

cleaned = cleaner.remove(population, Some(&reference));
# Row 0 should be removed.
assert len(cleaned) == 1;
np.testing.assert_array_equal(cleaned, np.array([10.0, 10.0, 10.0]));

```
