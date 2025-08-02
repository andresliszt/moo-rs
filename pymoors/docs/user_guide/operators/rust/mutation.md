A mutation operator in `moors` is any type that implements the {{ docs_rs("trait", "operators.MutationOperator") }} trait. For example:

```rust
use ndarray::ArrayViewMut1;
use crate::{operators::MutationOperator, random::RandomGenerator};

#[derive(Debug, Clone)]
/// Mutation operator that flips bits in a binary individual with a specified mutation rate.
pub struct BitFlipMutation {
    pub gene_mutation_rate: f64,
}

impl BitFlipMutation {
    pub fn new(gene_mutation_rate: f64) -> Self {
        Self { gene_mutation_rate }
    }
}

impl MutationOperator for BitFlipMutation {
    fn mutate<'a>(&self, mut individual: ArrayViewMut1<'a, f64>, rng: &mut impl RandomGenerator) {
        for gene in individual.iter_mut() {
            if rng.gen_bool(self.gene_mutation_rate) {
                *gene = if *gene == 0.0 { 1.0 } else { 0.0 };
            }
        }
    }
}
```

The main method to implement is `mutate`, which operates at the **individual** level using an `ndarray::ArrayViewMut1`. Predefined mutation operators include:

<table>
  <thead>
    <tr>
      <th>Mutation Operator</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href={{ docs_rs("struct", "operators.BitFlipMutation") }}>BitFlipMutation</a></td>
      <td>Randomly flips one or more bits in the binary representation, introducing small variations.</td>
    </tr>
    <tr>
      <td><a href="{{ docs_rs("struct", "operators.GaussianMutation") }}">GaussianMutation</a></td>
      <td>Adds Gaussian noise to each real-valued gene to locally explore the continuous solution space.</td>
    </tr>
    <tr>
      <td><a href="{{ docs_rs("struct", "operators.ScrambleMutation") }}">ScrambleMutation</a></td>
      <td>Selects a subsequence and randomly shuffles it, preserving the original elements but altering their order.</td>
    </tr>
    <tr>
      <td><a href="{{ docs_rs("struct", "operators.SwapMutation") }}">SwapMutation</a></td>
      <td>Swaps the positions of two randomly chosen genes to explore neighboring permutations.</td>
    </tr>
    <tr>
      <td><a href="{{ docs_rs("struct", "operators.DisplacementMutation") }}">DisplacementMutation</a></td>
      <td>Extracts a block of the permutation and inserts it at another position, preserving the block’s relative order.</td>
    </tr>
    <tr>
      <td><a href="{{ docs_rs("struct", "operators.UniformRealMutation") }}">UniformRealMutation</a></td>
      <td>Resets a real-valued gene based on a uniform distribution.</td>
    </tr>
    <tr>
      <td><a href="{{ docs_rs("struct", "operators.UniformBinaryMutation") }}">UniformRealMutation</a></td>
      <td> Resets a bit to a random 0 or 1 .</td>
    </tr>

  </tbody>
</table>
