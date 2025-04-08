# Genetic Operators

In pymoors, the genetic operators expected by the various algorithms are **mutation**, **crossover**, and a **sampler**. The selection and tournament operators are usually built into the algorithms; for example, NSGA2 employs selection and tournament procedures based on Rank and Crowding.

Currently, pymoors comes with a battery of **pre-defined genetic operators** implemented in Rust. The goal is to continuously add more classic genetic operators to reduce the amount of Python code that needs to be executed.

!!! warning
    At the moment, pymoors does not provide a way to define custom genetic operators using NumPy functions. This feature is planned to be available as soon as possible.

Each genetic operator in pymoors is exposed to Python as a class. For example, consider the following:

```python
from pymoors import RandomFloatSampling, GaussianMutation, ExponentialCrossover,

# Create a sampler that generates individuals randomly between 0 and 10.
sampler = RandomFloatSampling(min=0, max=10)
# Create a gauss mutator instance
mutation=GaussianMutation(gene_mutation_rate=0.1, sigma=0.01)
# Create an exponential crossover instance
crossover=ExponentialCrossover(exponential_crossover_rate = 0.75)
```

!!! warning
    1. Currently, these instances serve only as a means to inform Rust which operator to use, but they do not expose any public methods to Python; everything functions internally within the Rust core. We are currently evaluating whether it is necessary to expose these methods to Python so that users can interact with them.
    2. This section will be updated with more information on all the genetic operators in the near future.

## Check available python rust operators

`pymoors` provides a convenient method called `available_operators` that allows you to check which operators are implemented and exposed from Rust to Python. This includes operators for sampling, crossover, mutation and duplicates cleaner selection, and survival.

```python

from pymoors import available operators

>>> available_operators(operator_type = "mutation", include_docs = True)
{'BitFlipMutation': 'Mutation operator that flips bits in a binary individual with a specified mutation rate.', 'SwapMutation':  ...} # The dictionary was shortened for simplicity.

```

`operator_type` must be `'sampling'`, `'crossover'`, `'mutation'` or `'duplicates'`. Also the parameter `include_docs` includes the first line of the operator docstring, if it's set as `False` will return a list with class names only.
