## Mutation

A mutation operator in `moors` is any type that implements the `moors::MutationOperator` trait. For example:

```rust
use ndarray::ArrayViewMut1;
use crate::{operators::MutationOperator, random::RandomGenerator};

#[derive(Debug, Clone)]
/// Mutation operator that flips bits in a binary individual with a specified mutation rate.
pub struct BitFlipMutation {
    pub gene_mutation_rate: f64,
}

impl BitFlipMutation {
    #[allow(dead_code)]
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
      <td><a href="https://docs.rs/moors/latest/moors/operators/struct.BitFlipMutation.html">BitFlipMutation</a></td>
      <td>Randomly flips one or more bits in the binary representation, introducing small variations.</td>
    </tr>
    <tr>
      <td><a href="https://docs.rs/moors/latest/moors/operators/struct.GaussianMutation.html">GaussianMutation</a></td>
      <td>Adds Gaussian noise to each real-valued gene to locally explore the continuous solution space.</td>
    </tr>
    <tr>
      <td><a href="https://docs.rs/moors/latest/moors/operators/struct.ScrambleMutation.html">ScrambleMutation</a></td>
      <td>Selects a subsequence and randomly shuffles it, preserving the original elements but altering their order.</td>
    </tr>
    <tr>
      <td><a href="https://docs.rs/moors/latest/moors/operators/struct.SwapMutation.html">SwapMutation</a></td>
      <td>Swaps the positions of two randomly chosen genes to explore neighboring permutations.</td>
    </tr>
    <tr>
      <td><a href="https://docs.rs/moors/latest/moors/operators/struct.DisplacementMutation.html">DisplacementMutation</a></td>
      <td>Extracts a block of the permutation and inserts it at another position, preserving the block’s relative order.</td>
    </tr>
  </tbody>
</table>

## Crossover

A crossover operator in `moors` is any type that implements the `moors::CrossoverOperator` trait. For example:

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
      <td><a href="https://docs.rs/moors/latest/moors/operators/struct.ExponentialCrossover.html">ExponentialCrossover</a></td>
      <td>For Differential Evolution: starts at a random index and copies consecutive genes from the mutant vector while a uniform random number is below the crossover rate, then fills remaining positions from the target vector.</td>
    </tr>
    <tr>
      <td><a href="https://docs.rs/moors/latest/moors/operators/struct.OrderCrossover.html">OrderCrossover</a></td>
      <td>For permutations: copies a segment between two cut points from one parent, then fills the rest of the child with the remaining genes in the order they appear in the other parent.</td>
    </tr>
    <tr>
      <td><a href="https://docs.rs/moors/latest/moors/operators/struct.SimulatedBinaryCrossover.html">SimulatedBinaryCrossover</a></td>
      <td>For real-valued vectors: generates offspring by sampling each gene from a distribution centered on parent values, mimicking the spread of single-point binary crossover in continuous space.</td>
    </tr>
    <tr>
      <td><a href="https://docs.rs/moors/latest/moors/operators/struct.SinglePointBinaryCrossover.html">SinglePointBinaryCrossover</a></td>
      <td>For binary strings: selects one crossover point and swaps the tails of two parents at that point to produce two offspring.</td>
    </tr>
    <tr>
      <td><a href="https://docs.rs/moors/latest/moors/operators/struct.UniformBinaryCrossover.html">UniformBinaryCrossover</a></td>
      <td>For binary strings: for each bit position, randomly chooses which parent to inherit from (with a given probability), resulting in highly mixed offspring.</td>
    </tr>
  </tbody>
</table>

## Sampling

A sampling operator in `moors` is any type that implements the `moors::SamplingOperator` trait. For example:

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
      <td><a href="https://docs.rs/moors/latest/moors/operators/struct.RandomSamplingBinary.html">RandomSamplingBinary</a></td>
      <td>Generates a vector of random bits, sampling each position independently with equal probability.</td>
    </tr>
    <tr>
      <td><a href="https://docs.rs/moors/latest/moors/operators/struct.RandomSamplingFloat.html">RandomSamplingFloat</a></td>
      <td>Creates a real-valued vector by sampling each gene uniformly within specified bounds.</td>
    </tr>
    <tr>
      <td><a href="https://docs.rs/moors/latest/moors/operators/struct.RandomSamplingInt.html">RandomSamplingInt</a></td>
      <td>Produces an integer vector by sampling each gene uniformly from a given range.</td>
    </tr>
    <tr>
      <td><a href="https://docs.rs/moors/latest/moors/operators/struct.PermutationSampling.html">PermutationSampling</a></td>
      <td>Generates a random permutation by uniformly shuffling all indices.</td>
    </tr>
  </tbody>
</table>

## Selection Operator

The selection operator is a bit more restrictive, in that each pre‑defined algorithm in `moors` defines exactly one selection operator. For example, the `NSGA-II` algorithm uses a *ranking‑by‑crowding‑distance* selection operator, while `NSGA-III` uses a random selection operator. The user can only provide their own selection operator to a custom algorithm—not to the algorithms that come pre‑defined in moors.

A selection operator in `moors` is any type that implements the `moors::SelectionOperator` trait. For example:

```Rust
use crate::genetic::{D01, IndividualMOO};
use crate::operators::selection::{DuelResult, SelectionOperator};
use crate::random::RandomGenerator;

#[derive(Debug, Clone)]
pub struct RandomSelection;

impl SelectionOperator for RandomSelection {
    type FDim = ndarray::Ix2;

    fn tournament_duel<'a, ConstrDim>(
        &self,
        p1: &IndividualMOO<'a, ConstrDim>,
        p2: &IndividualMOO<'a, ConstrDim>,
        rng: &mut impl RandomGenerator,
    ) -> DuelResult
    where
        ConstrDim: D01,
    {
        if let result @ DuelResult::LeftWins | result @ DuelResult::RightWins =
            Self::feasibility_dominates(p1, p2)
        {
            return result;
        }
        // Otherwise, both are feasible or both are infeasible => random winner.
        if rng.gen_bool(0.5) {
            DuelResult::LeftWins
        } else {
            DuelResult::RightWins
        }
    }
}
```

Note that we have defined an associated type `type FDim = ndarray::Ix2`, this is because, in this example, this operator will be used for a multi‑objective algorithm. The selection operators defined in pymoors must specify the fitness dimension. Note that this is the selection operator used by the NSGA‑III algorithm: it performs a random selection that gives priority to feasibility, which is why we use the trait’s static method `Self::feasibility_dominates`.

## Survival Operator

The survival operator follows the same logic than selection operator, in that each pre‑defined algorithm in `moors` defines exactly one selection operator. For example, the `NSGA-II` algorithm uses a *ranking‑by‑crowding‑distance* survival operator, while `NSGA-III` uses a reference points based operator. The user can only provide their own survival operator to a custom algorithm—not to the algorithms that come pre‑defined in moors.

A survival operator in `moors` is any type that implements the `moors::SelectionOperator` trait. For example:

```Rust
use crate::genetic::{D01, IndividualMOO};
use crate::operators::selection::{DuelResult, SelectionOperator};
use crate::random::RandomGenerator;

#[derive(Debug, Clone)]
pub struct RandomSelection;

impl SelectionOperator for RandomSelection {
    type FDim = ndarray::Ix2;

    fn tournament_duel<'a, ConstrDim>(
        &self,
        p1: &IndividualMOO<'a, ConstrDim>,
        p2: &IndividualMOO<'a, ConstrDim>,
        rng: &mut impl RandomGenerator,
    ) -> DuelResult
    where
        ConstrDim: D01,
    {
        if let result @ DuelResult::LeftWins | result @ DuelResult::RightWins =
            Self::feasibility_dominates(p1, p2)
        {
            return result;
        }
        // Otherwise, both are feasible or both are infeasible => random winner.
        if rng.gen_bool(0.5) {
            DuelResult::LeftWins
        } else {
            DuelResult::RightWins
        }
    }
}
```

Note that we have defined an associated type `type FDim = ndarray::Ix2`, this is because, in this example, this operator will be used for a multi‑objective algorithm. The selection operators defined in pymoors must specify the fitness dimension. Note that this is the selection operator used by the NSGA‑III algorithm: it performs a random selection that gives priority to feasibility, which is why we use the trait’s static method `Self::feasibility_dominates`.
