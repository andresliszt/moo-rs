To use exact duplicates cleaner in `pymoors` just pass to the algorithm the cleaner object

```python
import numpy as np

from pymoors import ExactDuplicatesCleaner

cleaner = ExactDuplicatesCleaner()

raw_data = np.array([
    [1.0, 2.0, 3.0],  # row 0
    [4.0, 5.0, 6.0],  # row 1
    [1.0, 2.0, 3.0],  # row 2 (duplicate of row 0)
    [7.0, 8.0, 9.0],  # row 3
    [4.0, 5.0, 6.0],  # row 4 (duplicate of row 1)
])

cleaned = cleaner.remove(raw_data)

expected = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
])

np.testing.assert_array_equal(cleaned, expected)
```
