# Genetic Operators

In pymoors, the genetic operators expected by the various algorithms are **mutation**, **crossover**, and a **sampler**. The selection and tournament operators are usually built into the algorithms; for example, NSGA2 employs selection and tournament procedures based on Rank and Crowding.

## Built-in operators

pymoors comes with a battery of **pre-defined genetic operators** implemented in Rust. The goal is to continuously add more classic genetic operators to reduce the amount of Python code that needs to be executed.

Each genetic operator in pymoors is exposed to Python as a class. For example, consider the following:

```python
from pymoors import RandomSamplingFloat, GaussianMutation, ExponentialCrossover

# Create a sampler that generates individuals randomly between 0 and 10.
sampler = RandomSamplingFloat(min=0, max=10)
# Create a gauss mutator instance
mutation=GaussianMutation(gene_mutation_rate=0.5, sigma=0.1)
# Create an exponential crossover instance
crossover=ExponentialCrossover(exponential_crossover_rate = 0.75)
```

Each operator comes with a python method called `operate`.This method exposed to the Python side has the sole purpose of allowing the user to see how it is operating. Internally, it calls the operate method defined in the Rust side.

```python
# Call the sampler operate method (seed is Optional)
sample = sampler.operate(num_vars=2, population_size=10, seed=1)
>>> sample
array([[9.74244737, 6.91235611],
       [4.27974782, 1.7558605 ],
       ...
       [8.92289399, 3.90518075],
       [8.81965211, 0.42973335]])
```



### Check available python rust operators

`pymoors` provides a convenient method called `available_operators` that allows you to check which operators are implemented and exposed from Rust to Python. This includes operators for sampling, crossover, mutation and duplicates cleaner selection, and survival.

```python

from pymoors import available operators

>>> available_operators(operator_type = "mutation", include_docs = True)
{'BitFlipMutation': 'Mutation operator that flips bits in a binary individual with a specified mutation rate.', 'SwapMutation':  ...} # The dictionary was shortened for simplicity.

```

`operator_type` must be `'sampling'`, `'crossover'`, `'mutation'` or `'duplicates'`. Also the parameter `include_docs` includes the first line of the operator docstring, if it's set as `False` will return a list with class names only.

## Custom Operators

We allow the user to pass custom defined operators. They have to be a python class with a custom operate `method`, for example

```python

class CustomBitFlipMutation:
    def __init__(self, gene_mutation_rate: float = 0.5):
        self.gene_mutation_rate = gene_mutation_rate

    def operate(
        self,
        population: TwoDArray,
    ) -> TwoDArray:
        mask = np.random.random(population.shape) < self.gene_mutation_rate
        population[mask] = 1.0 - population[mask]
        return population

```

Note that is important that even working with binary data, the output array has to have float dtype (binary is encoded as 0.0 and 1.0). Everything in the Rust side is considered float, and we're studying if is worth to allow different dtypes: See this [issue](https://github.com/andresliszt/moo-rs/issues/172).

The signature of the operate methods are the following

```python

# Mutation
def operate(self, population: TwoDArray) -> TwoDArray: ...
# Crossover
def operate(self, parents_a: TwoDArray, parents_b: TwoDArray) -> TwoDArray:
# Sampling
def operate(self) -> TwoDArray:

```

Note that they work with `TwoDArray` (an alias for a numpy 2D array), meaning that they act at the population level. The crossover might the trickiest one, it's expected to be a binary crossover with 2 offsprings per each pair of parents, so the output in the crossover must be concatenation of the offsprings. It's important to follow the signatures and the float dtype, otherwise panic will be raised from `pyo3`. We're working to have a better error handling. See this [issue](https://github.com/andresliszt/moo-rs/issues/174)

!!! warning
       Random states introduced in the custom defined operators are not the same from the core in the Rust side. So to have reproducible results the user will have to pay attention on seeds in both: Python and Rust. See this [issue](https://github.com/andresliszt/moo-rs/issues/179)
