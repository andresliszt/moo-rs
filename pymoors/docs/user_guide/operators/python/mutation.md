A mutation operator in `pymoors` is just a class that defines the `operate` method:

```python
from pymoors.typing import TwoDArray

class BitFlipMutation:
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

`operate` acts at poblational level, as usual it means that it takes a 2D numpy array and returns 2D array too, where each row is the evaluation of a single individual.

There are many built-in mutation operators backed at the rust side

<table>
  <thead>
    <tr>
      <th>Mutation Operator</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>`pymoors.BitFlipMutation` </td>
      <td>Randomly flips one or more bits in the binary representation, introducing small variations.</td>
    </tr>
    <tr>
      <td>`pymoors.GaussianMutation`</td>
      <td>Adds Gaussian noise to each real-valued gene to locally explore the continuous solution space.</td>
    </tr>
    <tr>
      <td>`pymoors.ScrambleMutation` </td>
      <td>Selects a subsequence and randomly shuffles it, preserving the original elements but altering their order.</td>
    </tr>
    <tr>
      <td>`pymoors.SwapMutation` </td>
      <td>Swaps the positions of two randomly chosen genes to explore neighboring permutations.</td>
    </tr>
    <tr>
      <td>`pymoors.DisplacementMutation`</td>
      <td>Extracts a block of the permutation and inserts it at another position, preserving the block’s relative order.</td>
    </tr>
    <tr>
      <td>`pymoors.UniformRealMutation`</td>
      <td>Resets a real-valued gene based on a uniform distribution.</td>
    </tr>
    <tr>
      <td>`pymoors.UniformBinaryMutation`</td>
      <td> Resets a bit to a random 0 or 1 .</td>
    </tr>
  </tbody>
</table>

!!! info "`operate` at poblational level"
    in `moors` we allow the user to define the crossover at individual or poblational level, but in `pymoors` we force to
    work with poblational level. Technical reason is that in each user defined `operate` call we have to adquire python GIL
    in the rust side, poblational call requires just 1 call to the GIL.
