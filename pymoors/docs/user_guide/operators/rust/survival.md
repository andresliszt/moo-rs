
The survival operator follows the same logic than selection operator, in that each pre‑defined algorithm in `moors` defines exactly one selection operator. For example, the `NSGA-II` algorithm uses a *ranking‑by‑crowding‑distance* survival operator, while `NSGA-III` uses a reference points based operator.

First, determine whether your algorithm uses ranking. If it does, implement the trait: {{ docs_rs("trait", "operators.survival.moo.FrontsAndRankingBasedSurvival") }}. If not, implement: {{ docs_rs("trait", "operators.survival.moo.SurvivalOperator") }}

We use [NSGA-II](../../algorithms/nsga2.md) as an example, which uses both ranking and survival scoring.

```rust
use ndarray::{Array1, Array2};

use crate::{
    genetic::{D12, Fronts},
    operators::survival::moo::FrontsAndRankingBasedSurvival,
    random::RandomGenerator,
};

#[derive(Debug, Clone, Default)]
pub struct Nsga2RankCrowdingSurvival;

impl Nsga2RankCrowdingSurvival {
    pub fn new() -> Self {
        Self {}
    }
}

impl FrontsAndRankingBasedSurvival for Nsga2RankCrowdingSurvival {
    fn set_front_survival_score<ConstrDim>(
        &self,
        fronts: &mut Fronts<ConstrDim>,
        _rng: &mut impl RandomGenerator,
    ) where
        ConstrDim: D12,
    {
        for front in fronts.iter_mut() {
            let crowding_distance = crowding_distance(&front.fitness);
            front.set_survival_score(crowding_distance);
        }
    }
}

/// Computes the crowding distance for a given Pareto population_fitness.
///
/// # Parameters:
/// - `population_fitness`: A 2D array where each row represents an individual's fitness values.
///
/// # Returns:
/// - A 1D array of crowding distances for each individual in the population_fitness.
fn crowding_distance(population_fitness: &Array2<f64>) -> Array1<f64> {
    // Omitted for simplicity
}
```

### Notes on Fronts and Traits

- `Fronts` is a type alias for a `Vec` of {{ docs_rs("type", "genetic.PopulationMOO") }}.
- `PopulationMOO` is a container for individuals, including `genes`, `fitness`, `constraints` (if any), `rank` (if any), and `survival_score` (if any).
- Fronts are sorted by dominance (0-dominated first, then 1-dominated, etc.).
- {{ docs_rs("type", "genetic.D12") }} is a trait for constraint output dimensions, allowing flexibility between `Array1` and `Array2`.


In the `moors` framework, implementing a survival strategy for multi-objective optimization algorithms involves defining the trait {{ docs_rs("trait", "operators.survival.moo.FrontsAndRankingBasedSurvival", tymethod = "set_front_survival_score", label = "set_front_survival_score") }}. `set_front_survival_score` method receives a vector of populations (`Vec<PopulationMOO>`), where each element represents a Pareto front. Each front contains:

- `genes`: An `Array2<f64>` representing the genetic encoding of individuals.
- `fitness`: An `Array2<f64>` where each row corresponds to the fitness values of an individual.
- `rank`: An `Array1<f64>` indicating the dominance rank of each individual.

The purpose of `set_front_survival_score` is to assign a **survival score** to each individual within a front. In the provided example, this score is computed using the `crowding_distance` function, which returns an `Array1<f64>` where each element corresponds to the survival score of an individual.

After survival scores are assigned, the {{ docs_rs("trait", "operators.survival.moo.FrontsAndRankingBasedSurvival", method = "operate", label = "operate") }} method is responsible for selecting the individuals that will survive to the next generation.

The splitting front approach is used for selecting the survivors for the next generation, by default `moors` will prefer an individual with higher survival score, you can modify this by overwritting the method selection `Minimize` enum member

```Rust
fn scoring_comparison(&self) -> SurvivalScoringComparison {
    SurvivalScoringComparison::Minimize
}
```
