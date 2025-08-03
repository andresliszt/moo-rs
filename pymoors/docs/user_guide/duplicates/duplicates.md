
# Duplicates Cleaner

In genetic algorithms, one way to maintain diversity is to eliminate duplicates generation after generation. This operation can be computationally expensive but is often essential to ensure that the algorithm continues to explore new individuals that can improve the objectives.

!!! info
    Benchmarks comparing pymoors/moors with other Python/Rust frameworks will be published soon. These benchmarks will highlight the importance and performance impact of duplicate elimination in genetic algorithms.

=== "Rust"
    {% include-markdown "user_guide/duplicates/rust/custom.md" %}

=== "Python"
    {% include-markdown "user_guide/duplicates/python/custom.md" %}


## Exact Duplicates Cleaner

Based on exact elimination, meaning that two individuals (genes1 and genes2) are considered duplicates if and only if each element in genes1 is equal to the corresponding element in genes2. Internally, this cleaner uses Rust’s HashSet.

=== "Rust"
    {% include-markdown "user_guide/duplicates/rust/exact.md" %}

=== "Python"
    {% include-markdown "user_guide/duplicates/python/exact.md" %}

## Close Duplicates Cleaner

This is designed for real-valued problems where two individuals are very similar, but not exactly identical. In such cases, it is beneficial to consider them as duplicates based on a proximity metric—typically the Euclidean distance. This cleaner implements a cleaner of this style that uses the square of the Euclidean distance between two individuals and considers them duplicates if this value is below a configurable tolerance (epsilon).

=== "Rust"
    {% include-markdown "user_guide/duplicates/rust/close.md" %}

=== "Python"
    {% include-markdown "user_guide/duplicates/python/close.md" %}

!!! Danger "Caution"

     This duplicate elimination algorithm can be computationally expensive when the population size and the number of offsprings are large, because it requires calculating the distance matrix among offsprings first, and then between offsprings and the current population to ensure duplicate elimination using this criterion. The algorithm has a complexity of O(n*m) where n is the population size and m is the number of offsprings.
