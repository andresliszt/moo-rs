A sampling operator in `pymoors` is just a class that defines the `operate` method:

```python
from pymoors.typing import TwoDArray

class RandomSamplingBinary:
    def operate(self, population: TwoDArray) -> TwoDArray:
        mask = np.random.random(population.shape) < 0.5
        return mask.astype(np.float64)

```

`operate` acts at poblational level, as usual it means that it takes a 2D numpy array and returns 2D array too, where each row is a sampled individual.

<table>
  <thead>
    <tr>
      <th>Sampling Operator</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>`pymoors.RandomSamplingBinary`</td>
      <td>Generates a vector of random bits, sampling each position independently with equal probability.</td>
    </tr>
    <tr>
      <td>`pymoors.RandomSamplingFloat`</td>
      <td>Creates a real-valued vector by sampling each gene uniformly within specified bounds.</td>
    </tr>
    <tr>
      <td>`pymoors.RandomSamplingInt`</td>
      <td>Produces an integer vector by sampling each gene uniformly from a given range.</td>
    </tr>
    <tr>
      <td>`pymoors.PermutationSampling`</td>
      <td>Generates a random permutation by uniformly shuffling all indices.</td>
    </tr>
  </tbody>
</table>
