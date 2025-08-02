A crossover operator in `moors` is any type that implements the {{ docs_rs("trait", "operators.CrossoverOperator") }} trait. For example:

```rust
use ndarray::{Array1, Axis, concatenate, s};
use crate::operators::CrossoverOperator;
use crate::random::RandomGenerator;

#[derive(Debug, Clone)]
/// Single-point crossover operator for binary-encoded individuals.
pub struct SinglePointBinaryCrossover;

impl CrossoverOperator for SinglePointBinaryCrossover {
    fn crossover(
        &self,
        parent_a: &Array1<f64>,
        parent_b: &Array1<f64>,
        rng: &mut impl RandomGenerator,
    ) -> (Array1<f64>, Array1<f64>) {
        let num_genes = parent_a.len();
        // Select a crossover point between 1 and num_genes - 1
        let crossover_point = rng.gen_range_usize(1, num_genes);
        // Split parents at the crossover point and create offspring
        let offspring_a = concatenate![
            Axis(0),
            parent_a.slice(s![..crossover_point]),
            parent_b.slice(s![crossover_point..])
        ];
        let offspring_b = concatenate![
            Axis(0),
            parent_b.slice(s![..crossover_point]),
            parent_a.slice(s![crossover_point..])
        ];
        (offspring_a, offspring_b)
    }
}
```

The main method to implement is `crossover`, which takes two parents (`ndarray::Array1`) and produces two offspring. Predefined crossover operators include:

<table>
  <thead>
    <tr>
      <th>Crossover Operator</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href={{ docs_rs("struct", "operators.ExponentialCrossover") }}>ExponentialCrossover</a></td>
      <td>For Differential Evolution: starts at a random index and copies consecutive genes from the mutant vector while a uniform random number is below the crossover rate, then fills remaining positions from the target vector.</td>
    </tr>
    <tr>
      <td><a href={{ docs_rs("struct", "operators.OrderCrossover") }}>OrderCrossover</a></td>
      <td>For permutations: copies a segment between two cut points from one parent, then fills the rest of the child with the remaining genes in the order they appear in the other parent.</td>
    </tr>
    <tr>
      <td><a href={{ docs_rs("struct", "operators.SimulatedBinaryCrossover") }}>SimulatedBinaryCrossover</a></td>
      <td>For real-valued vectors: generates offspring by sampling each gene from a distribution centered on parent values, mimicking the spread of single-point binary crossover in continuous space.</td>
    </tr>
    <tr>
      <td><a href={{ docs_rs("struct", "operators.SinglePointBinaryCrossover") }}>SinglePointBinaryCrossover</a></td>
      <td>Selects one crossover point and swaps the tails of two parents at that point to produce two offspring.</td>
    </tr>
    <tr>
      <td><a href={{ docs_rs("struct", "operators.UniformBinaryCrossover") }}>UniformBinaryCrossover</a></td>
      <td>For each bit position, randomly chooses which parent to inherit from (with a given probability), resulting in highly mixed offspring.</td>
    </tr>
    <tr>
      <td><a href={{ docs_rs("struct", "operators.TwoPointBinaryCrossover") }}>TwoPointBinaryCrossover</a></td>
      <td>Exchanges segments between two parents at two randomly chosen points to create offspring.</td>
    </tr>
    <tr>
      <td><a href={{ docs_rs("struct", "operators.ArithmeticCrossover") }}>ArithmeticCrossover</a></td>
      <td>Exchanges segments between two parents at two randomly chosen points to create offspring.</td>
    </tr>
  </tbody>
</table>
