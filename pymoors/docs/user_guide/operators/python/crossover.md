A crossover operator in `pymoors` is just a class that defines the `operate` method:

```python
from pymoors.typing import TwoDArray

class SinglePointBinaryCrossover:
    def operate(
        self,
        parents_a: TwoDArray,
        parents_b: TwoDArray,
    ) -> TwoDArray:
        n_pairs, n_genes = parents_a.shape
        offsprings = np.empty((2 * n_pairs, n_genes), dtype=parents_a.dtype)
        for i in range(n_pairs):
            a = parents_a[i]
            b = parents_b[i]
            point = np.random.randint(1, n_genes)
            c1 = np.concatenate((a[:point], b[point:]))
            c2 = np.concatenate((b[:point], a[point:]))
            offsprings[2 * i] = c1
            offsprings[2 * i + 1] = c2
        return offsprings
```

`operate` acts at poblational level, as usual it means that it takes **two parents** as 2D numpy arrays and returns a **single** 2D array of length twice the number of crossovers (two children per crossover).

There are many built-in crossover operators backed at the rust side

<table>
  <thead>
    <tr>
      <th>Crossover Operator</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>`pymoors.ExponentialCrossover`</td>
      <td>For Differential Evolution: starts at a random index and copies consecutive genes from the mutant vector while a uniform random number is below the crossover rate, then fills remaining positions from the target vector.</td>
    </tr>
    <tr>
      <td>`pymoors.OrderCrossover`</td>
      <td>For permutations: copies a segment between two cut points from one parent, then fills the rest of the child with the remaining genes in the order they appear in the other parent.</td>
    </tr>
    <tr>
      <td>`pymoors.SimulatedBinaryCrossover`</td>
      <td>For real-valued vectors: generates offspring by sampling each gene from a distribution centered on parent values, mimicking the spread of single-point binary crossover in continuous space.</td>
    </tr>
    <tr>
      <td>`pymoors.SinglePointBinaryCrossover`</td>
      <td>Selects one crossover point and swaps the tails of two parents at that point to produce two offspring.</td>
    </tr>
    <tr>
      <td>`pymoors.UniformBinaryCrossover`</td>
      <td>For each bit position, randomly chooses which parent to inherit from (with a given probability), resulting in highly mixed offspring.</td>
    </tr>
    <tr>
      <td>`pymoors.TwoPointBinaryCrossover`</td>
      <td>Exchanges segments between two parents at two randomly chosen points to create offspring.</td>
    </tr>
    <tr>
      <td>`pymoors.ArithmeticCrossover`</td>
      <td>Generates offspring by computing a weighted average of two parent solutions (for each gene, child = α·parent₁ + (1−α)·parent₂).</td>
    </tr>
  </tbody>
</table>

!!! info "`operate` at poblational level"
    in `moors` we allow the user to define the crossover at individual or poblational level, but in `pymoors` we force to
    work with poblational level. Technical reason is that in each user defined `operate` call we have to adquire python GIL
    in the rust side, poblational call requires just 1 call to the GIL.
