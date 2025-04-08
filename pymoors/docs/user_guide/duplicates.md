# Duplicates Cleaner

In genetic algorithms, one way to maintain diversity is to eliminate duplicates generation after generation. This operation can be computationally expensive but is often essential to ensure that the algorithm continues to explore new individuals that can improve the objectives. Currently, pymoors includes two pre-defined duplicate cleaners.

## Exact Duplicates Cleaner

Based on exact elimination, meaning that two individuals (genes1 and genes2) are considered duplicates if and only if each element in genes1 is equal to the corresponding element in genes2. Internally, this cleaner uses Rust’s HashSet, which operates **extremely fast** for duplicate elimination.

!!! warning
    Benchmarks comparing pymoors with other Python libraries will be published soon. These benchmarks will highlight the importance and performance impact of duplicate elimination in genetic algorithms.

```python
from pymoors import ExactDuplicatesCleaner

cleaner = ExactDuplicatesCleaner()

```


## CloseDuplicatesCleaner

CloseDuplicatesCleaner is designed for real-valued problems where two individuals are very similar, but not exactly identical. In such cases, it is beneficial to consider them as duplicates based on a proximity metric—typically the Euclidean distance. pymoors implements a cleaner of this style that uses the square of the Euclidean distance between two individuals and considers them duplicates if this value is below a configurable tolerance (epsilon).

```python
from pymoors import CloseDuplicatesCleaner

cleaner = CloseDuplicatesCleaner(epsilon=1e-5)

```

!!! Danger "Caution"

     This duplicate elimination algorithm can be computationally expensive when the population size and the number of offsprings are large, because it requires calculating the distance matrix among offsprings first, and then between offsprings and the current population to ensure duplicate elimination using this criterion. The algorithm has a complexity of O(n*m) where n is the population size and m is the number of offsprings.
