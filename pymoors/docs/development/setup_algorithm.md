# Genetic Algorithm Setup

**Procedure:** In `moors`, genetic algorithms follow a [structured approach](../user_guide/algorithms/introduction/introduction.md#algorithms) that involves three families of operators: **mating**, **selection**, and **survival**.

## Mating Operators (Shared Across Algorithms)
These operators are modular and can be used with any algorithm:

- **Sampling**: Generates the initial population (e.g., random, Latin hypercube).
- **Crossover**: Recombines parents to produce offspring.
- **Mutation**: Perturbs offspring to maintain diversity and exploration.

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

Creating a new algorithm in `moors` is relatively simple. We use [NSGA-II](../user_guide/algorithms/nsga2.md) and [NSGA-III](../user_guide/algorithms/nsga3.md) as examples. The user must define the specific survival and selection operators.

### Survival and Selection Operators Without Arguments

The following example shows the implementation of the `NSGA-II` algorithm using the macro `create_algorithm_and_builder`:

```Rust
use crate::{
    define_algorithm_and_builder,
    operators::{
        selection::moo::Nsga2RankAndScoringSelection, survival::moo::Nsga2RankCrowdingSurvival,
    },
};

define_algorithm_and_builder!(
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
    Nsga2, Nsga2RankAndScoringSelection, Nsga2RankCrowdingSurvival
);
```

This macro generates two structs: `Nsga2` and `Nsga2Builder`. The new struct `Nsga2` is a type alias of the core `GeneticAlgorithm` struct, fixing the generic selection/survival operators to `RankAndScoringSelection` and `Nsga2RankCrowdingSurvival` respectively. The `Nsga2Builder` is the builder struct that uses the [Rust builder design pattern](https://rust-unofficial.github.io/patterns/patterns/creational/builder.html) to create `Nsga2` instances.

Internally, `Nsga2Builder` wraps the {{ docs_rs("struct", "algorithms.AlgorithmBuilder") }} as shown below:

```Rust
pub struct Nsga2Builder<S, Cross, Mut, F, G, DC>
where
    S: SamplingOperator,
    Nsga2RankAndScoringSelection: SelectionOperator<FDim = F::Dim>,
    Nsga2RankCrowdingSurvival: SurvivalOperator<FDim = F::Dim>,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: FitnessFn,
    G: ConstraintsFn,
    DC: PopulationCleaner,
{
    inner: AlgorithmBuilder<S, Nsga2RankAndScoringSelection, Nsga2RankCrowdingSurvival, Cross, Mut, F, G, DC>,
}
```

This struct delegates all builder methods to the inner builder, which was created using the [derive builder crate](https://crates.io/crates/derive_builder).

### Survival and Selection Operators With Arguments

When either survival or selection operators take arguments, the macro needs to know about them. The following example shows the implementation of the `NSGA-III` algorithm. The survival operator takes two arguments: `reference_points` and the boolean `are_aspirational`. For more information, refer to the [algorithm docs](../user_guide/algorithms/nsga3.md).

```Rust
use ndarray::Array2;

use crate::{
    define_algorithm_and_builder,
    operators::{
        selection::moo::Nsga3RandomSelection,
        survival::moo::Nsga3ReferencePointsSurvival,
    },
};

define_algorithm_and_builder!(
    /// NSGA-III algorithm wrapper.
    ///
    /// This struct is a thin facade over [`GeneticAlgorithm`] preset with
    /// the NSGA-III survival and selection strategy.
    ///
    /// * **Selection:** [`RandomSelection`]
    /// * **Survival:**  [`Nsga3ReferencePointsSurvival`] (elitist, reference-point based)
    ///
    /// Construct it with [`Nsga3Builder`](crate::algorithms::Nsga3Builder).
    /// After building, call [`run`](GeneticAlgorithm::run)
    /// and then [`population`](GeneticAlgorithm::population) to retrieve the
    /// final non-dominated set.
    ///
    /// For algorithmic details, see:
    /// Kalyanmoy Deb and Himanshu Jain (2014),
    /// "An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based
    /// Nondominated Sorting Approach, Part I: Solving Problems with Box Constraints",
    /// *IEEE Transactions on Evolutionary Computation*, vol. 18, no. 4,
    /// pp. 577–601, Aug. 2014.
    /// DOI: 10.1109/TEVC.2013.2281535
    Nsga3,
    Nsga3RandomSelection,
    Nsga3ReferencePointsSurvival,
    survival_args = [ reference_points: Array2<f64>, are_aspirational: bool ]
);
```

In this case, `Nsga3Builder` extends the {{ docs_rs("struct", "algorithms.AlgorithmBuilder") }} setters by adding `reference_points` and `are_aspirational`, which are passed to the survival operator in the build method.

Similarly, if the selection operator takes extra arguments, use `selection_args` in the macro.

### Writing a New Survival Operator and Selector Operator

Refer to the [survival section](../user_guide/operators/operators.md#survival) and [selection section](../user_guide/operators/operators.md#selection) in the user guide.
