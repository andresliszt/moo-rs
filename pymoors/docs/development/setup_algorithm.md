
# Setting Up Genetic Algorithms

**Procedure:** In `moors`, genetic algorithms follow a [structured approach](../user_guide/algorithms/introduction/introduction.md#algorithms) involving three families of operators: **mating**, **selection**, and **survival**.

## Mating Operators (Shared Across Algorithms)
These operators are modular and can be used with any algorithm:

- **Sampling**: Generates the initial population (e.g., random, Latin hypercube).
- **Crossover**: Recombines parents to produce offspring.
- **Mutation**: Perturbs offspring to maintain diversity and exploration.

## What Differentiates Algorithms
- **Selection** and **Survival** are algorithm-specific components.
- **Survival** is typically the most distinctive part, as it determines which individuals proceed to the next generation.

## Ranking & Survival Scoring

### Ranking (Dominance-Based)
- **Dominance**: Solution *A* dominates *B* if *A* is no worse in all objectives and strictly better in at least one.
- **Non-dominated (0-dominated)**: Dominated by **zero** other solutions → **Front 1**.
- **1-dominated, 2-dominated, …**: Dominated by exactly 1, 2, … other solutions. After removing Front 1, the next non-dominated set forms **Front 2**, and so on (Pareto fronts).

Note: **Not all algorithms use ranking**. For example, [NSGA-III](../user_guide/algorithms/nsga3.md) does not.

### Survival Scoring
Survival scoring is a numerical measure used to evaluate individuals. In rank-based algorithms like [NSGA-II](../user_guide/algorithms/nsga2.md), it serves as a tie-breaker within a front. When candidates share the same front rank, a **survival score** helps preserve quality and diversity—commonly using crowding distance.

In algorithms that do not use ranking, this score can be used to select the top $N$ survivors in each generation. For instance, [IBEA](../user_guide/algorithms/ibea.md) uses the hypervolume indicator, preferring individual $x$ over $y$ if $x$ contributes more to the hypervolume.

## Writing a New Algorithm in `moors`

To create a new algorithm, define a new survival struct and optionally a new selection operator. This is optional because `moors` already provides two widely used selection operators:

- {{ docs_rs("struct", "operators.selection.moo.RandomSelection") }}
- {{ docs_rs("struct", "operators.selection.moo.TournamentSelection") }}

All existing algorithms in `moors` use one of these. Nevertheless, we explain how to define a new one.


Creating a new algorithm is relatively simple in `moors`, We use [NSGA-II](../user_guide/algorithms/nsga2.md) and as [NSGA-III](../user_guide/algorithms/nsga3.md) an example.


```Rust
use crate::{
    create_algorithm_and_builder,
    operators::{
        selection::moo::RankAndScoringSelection, survival::moo::Nsga2RankCrowdingSurvival,
    },
};

create_algorithm_and_builder!(
    /// NSGA-II algorithm wrapper.
    ///
    /// This struct is a thin facade over [`GeneticAlgorithm`] preset with
    /// the NSGA-II survival and selection strategy.
    ///
    /// * **Selection:** [`RankAndScoringSelection`]
    /// * **Survival:**  [`Nsga2RankCrowdingSurvival`] (elitist, crowding-distance)
    ///
    /// Construct it with [`Nsga2Builder`](crate::algorithms::Nsga2Builder).
    /// After building, call [`run`](GeneticAlgorithm::run)
    /// and then [`population`](GeneticAlgorithm::population) to retrieve the
    /// final non-dominated set.
    ///
    /// For algorithmic details, see:
    /// Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and T. Meyarivan (2002),
    /// "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II",
    /// *IEEE Transactions on Evolutionary Computation*, vol. 6, no. 2,
    /// pp. 182–197, Apr. 2002.
    /// DOI: 10.1109/4235.996017
    Nsga2, RankAndScoringSelection, Nsga2RankCrowdingSurvival
);

```

`create_algorithm_and_builder` generates two structs: `Nsga2` and `Nsga2Builder`



### Writing a New Survival Operator

First, determine whether your algorithm uses ranking. If it does, implement the trait: {{ docs_rs("trait", "operators.survival.moo.FrontsAndRankingBasedSurvival") }}. If not, implement: {{ docs_rs("trait", "operators.survival.moo.SurvivalOperator") }}

We use [NSGA-II](../user_guide/algorithms/nsga2.md) as an example, which uses both ranking and survival scoring.

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
