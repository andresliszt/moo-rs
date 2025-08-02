A sampling operator in `moors` is any type that implements the {{ docs_rs("trait", "operators.SamplingOperator") }} trait. For example:

```rust
use ndarray::Array1;
use crate::{operators::SamplingOperator, random::RandomGenerator};

/// Sampling operator for binary variables.
#[derive(Debug, Clone)]
pub struct RandomSamplingBinary;

impl SamplingOperator for RandomSamplingBinary {
    fn sample_individual(&self, num_vars: usize, rng: &mut impl RandomGenerator) -> Array1<f64> {
        (0..num_vars)
            .map(|_| if rng.gen_bool(0.5) { 1.0 } else { 0.0 })
            .collect()
    }
}
```

The main method to implement is `sample_individual`, which produces an **individual** as an `ndarray::Array1`. Predefined sampling operators include:

<table>
  <thead>
    <tr>
      <th>Sampling Operator</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href={{ docs_rs("struct", "operators.RandomSamplingBinary") }}>RandomSamplingBinary</a></td>
      <td>Generates a vector of random bits, sampling each position independently with equal probability.</td>
    </tr>
    <tr>
      <td><a href={{ docs_rs("struct", "operators.RandomSamplingFloat") }}>RandomSamplingFloat</a></td>
      <td>Creates a real-valued vector by sampling each gene uniformly within specified bounds.</td>
    </tr>
    <tr>
      <td><a href={{ docs_rs("struct", "operators.RandomSamplingInt") }}>RandomSamplingInt</a></td>
      <td>Produces an integer vector by sampling each gene uniformly from a given range.</td>
    </tr>
    <tr>
      <td><a href={{ docs_rs("struct", "operators.PermutationSampling") }}>PermutationSampling</a></td>
      <td>Generates a random permutation by uniformly shuffling all indices.</td>
    </tr>
  </tbody>
</table>
