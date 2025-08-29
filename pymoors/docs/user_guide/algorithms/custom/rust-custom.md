
In the [introduction](../introduction/introduction.md#algorithms), an example with a pre-set genetic algorithm was shown, the famous `NSGA-II`; in that particular algorithm, the [selection](../../operators/operators.md#selection) and [survival](../../operators/operators.md#survival) operators were fixed at {{ docs_rs("struct", "operators.selection.moo.RankAndScoringSelection") }} and {{ docs_rs("struct", "operators.survival.moo.nsga2.Nsga2RankCrowdingSurvival") }} respectively, hence the user can only vary the `sampling`, `mutation`, `crossover`, and duplicates cleaner operators. In `moors`, there is the possibility to build your own genetic algorithms, meaning the user can generate the respective `selection` and `survival` operators.

The way to define a custom algorithm

```Rust
use crate::{
    algorithms::{AlgorithmBuilder, AlgorithmError},
    duplicates::NoDuplicatesCleaner,
    genetic::{Individual, Population},
    helpers::dimension::{D01, D12, Dimension},
    operators::{DuelResult, SelectionOperator, SurvivalOperator},
    // Bring your concrete sampling/crossover/mutation and fitness here:
    operators::{CrossoverOperator, MutationOperator, SamplingOperator},
    random::RandomGenerator,
};

/// ----------------
/// Custom Selection
/// ----------------
#[derive(Clone, Debug)]
pub struct MyCustomSelection;

impl SelectionOperator for MyCustomSelection
{
    type FDim = ndarray::Ix2; // If your algorithm is 1D use type FDim = ndarray::Ix1;

    /// Tournament between 2 individuals.
    fn tournament_duel<'a, ConstrDim>(
        &self,
        p1: &Individual<'a, <Self::FDim as Dimension>::Smaller, ConstrDim>,
        p2: &Individual<'a, <Self::FDim as Dimension>::Smaller, ConstrDim>,
        rng: &mut impl RandomGenerator,
    ) -> DuelResult
    where
        <Self::FDim as Dimension>::Smaller: D01,
        ConstrDim: D01,
    {
        todo!("return a DuelResult according to your policy")
    }
}

/// ---------------
/// Custom Survival
/// ---------------
#[derive(Clone, Debug)]
pub struct MyCustomSurvival;

impl SurvivalOperator for MyCustomSurvival
{
    type FDim = ndarray::Ix2; // If your algorithm is 1D use type FDim = ndarray::Ix1;

    fn operate<ConstrDim>(
        &mut self,
        population: Population<Self::FDim, ConstrDim>,
        num_survive: usize,
        rng: &mut impl RandomGenerator,
    ) -> Population<Self::FDim, ConstrDim>
    where
        ConstrDim: D12,
    {
        todo!("return the reduced population with exactly `num_survive` survivors")
    }
}

/// ------------------------------
/// Using the custom operators in the builder
/// ------------------------------
fn main() -> Result<(), AlgorithmError> {
    let custom_ga = AlgorithmBuilder::default()
        // Provide your concrete operators and functions below:
        .sampler(/* your SamplingOperator */)
        .selector(MyCustomSelection)
        .survivor(MyCustomSurvival)
        .crossover(/* your CrossoverOperator */)
        .mutation(/* your MutationOperator */)
        .duplicates_cleaner(/* your DuplicatesCleaner */)
        .fitness_fn(/* your fitness */)
        .constraints_fn(/* your constraints */)
        .num_vars(/* number of decision variables */)
        .population_size(/* μ */)
        .num_offsprings(/* λ */)
        .num_iterations(/* iterations */)
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .keep_infeasible(false)
        .verbose(true)
        .build()?;

    custom_ga.run()?;

    Ok(())
}
```

!!! info
    You have noticed that we use `type FDim = ndarray::Ix2` or `type FDim = ndarray::Ix1` in the definition of the `survival` and `selection` operators. These are associated types that must match the array type returned by your fitness function, which can be 1D for single-objective algorithms or 2D for multi-objective ones.
