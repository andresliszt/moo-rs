













```python
import numpy as np

from pymoors import (
    Nsga2,
    RandomSamplingBinary,
    BitFlipMutation,
    SinglePointBinaryCrossover,
    ExactDuplicatesCleaner,
)
from pymoors.typing import TwoDArray


PROFITS = np.array([2, 3, 6, 1, 4])
QUALITIES = np.array([5, 2, 1, 6, 4])
WEIGHTS = np.array([2, 3, 6, 2, 3])
CAPACITY = 7


def knapsack_fitness(genes: TwoDArray) -> TwoDArray:
    # Calculate total profit
    profit_sum = np.sum(PROFITS * genes, axis=1, keepdims=True)
    # Calculate total quality
    quality_sum = np.sum(QUALITIES * genes, axis=1, keepdims=True)

    # We want to maximize profit and quality,
    # so in pymoors we minimize the negative values
    f1 = -profit_sum
    f2 = -quality_sum
    return np.column_stack([f1, f2])


def knapsack_constraint(genes: TwoDArray) -> TwoDArray:
    # Calculate total weight
    weight_sum = np.sum(WEIGHTS * genes, axis=1, keepdims=True)
    # Inequality constraint: weight_sum <= capacity
    return weight_sum - CAPACITY


algorithm = Nsga2(
    sampler=RandomSamplingBinary(),
    crossover=SinglePointBinaryCrossover(),
    mutation=BitFlipMutation(gene_mutation_rate=0.5),
    fitness_fn=knapsack_fitness,
    constraints_fn=knapsack_constraint,
    duplicates_cleaner=ExactDuplicatesCleaner(),
    num_vars=5,
    population_size=16,
    num_offsprings=16,
    num_iterations=10,
    mutation_rate=0.1,
    crossover_rate=0.9,
    keep_infeasible=False,
    verbose=False,
)

algorithm.run()
```



In this **small example**, the algorithm finds a **single** solution on the Pareto front: selecting the items **(A, D, E)**, with a profit of **7** and a quality of **15**. This means there is no other combination that can match or exceed *both* objectives without exceeding the knapsack capacity (7).

Once the algorithm finishes, it stores a `population` attribute that contains all the individuals evaluated during the search.
























































```python
>>> population = algorithm.population
# Get genes
>>> population.genes
array([[1., 0., 0., 1., 1.],
       [1., 1., 0., 1., 0.],
       [0., 1., 0., 0., 1.],
       [1., 0., 0., 0., 1.],
       [1., 0., 0., 1., 0.],
       [0., 0., 0., 1., 1.],
       [0., 0., 1., 0., 0.],
       [1., 1., 0., 0., 0.],
       [0., 1., 0., 1., 0.],
       [1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1.],
       [0., 0., 0., 1., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0.]])

```



















































```python
# Get fitness
>>> population.fitness
array([[ -7., -15.],
       [ -6., -13.],
       [ -7.,  -6.],
       [ -6.,  -9.],
       [ -3., -11.],
       [ -5., -10.],
       [ -6.,  -1.],
       [ -5.,  -7.],
       [ -4.,  -8.],
       [ -2.,  -5.],
       [ -4.,  -4.],
       [ -1.,  -6.],
       [ -3.,  -2.],
       [ -0.,  -0.]])

```



















































```python
# Get constraints
>>> population.constraints
array([[ 0.],
       [ 0.],
       [-1.],
       [-2.],
       [-3.],
       [-2.],
       [-1.],
       [-2.],
       [-2.],
       [-5.],
       [-4.],
       [-5.],
       [-4.],
       [-7.]])

```



















































```python
# Get rank (for Nsga2)
>>> population.rank
array([0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6], dtype=uint64)

```



















































```python
>>> best = population.best
>>> best
[<pymoors.schemas.Individual at 0x1c6699386e0>]

```













































```python
>>> best[0].genes
array([1., 0., 0., 1., 1.])

```













































```python
>>> best[0].fitness
array([ -7., -15.])

```


> **ℹ️ Note – Population Size and Duplicates**
>
> Note that although the specified `population_size` was 16, the final population ended up
> being 13 individuals, of which 1 had `rank = 0`.
> This is because we used the `keep_infeasible=False` argument, removing any individual
> that did not satisfy the constraints_fn (in this case, the weight constraint).
> We also used a duplicate remover called `ExactDuplicatesCleaner` that eliminates all exact
> duplicates—meaning whenever `genes1 == genes2` in every component.
>
> **💡 Tip – Variable Types in pymoors**
>
> In **pymoors**, there is no strict enforcement of whether variables are integer, binary, or
> real. The core Rust implementation works with `f64` ndarrays.
> To preserve a specific variable type—binary, integer, or real—you must ensure that the
> genetic operators themselves maintain it.
>
> It is **the user's responsibility** to choose the appropriate genetic operators for the
> variable type in question. In the knapsack example, we use binary-style genetic operators,
> which is why the solutions are arrays of 0 s and 1 s.
