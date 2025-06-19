//! # NSGA‑II – Fast Elitist Multi‑Objective GA
//!
//! This implementation follows the seminal paper
//! **K. Deb, A. Pratap, S. Agarwal & T. Meyarivan,
//! “A Fast and Elitist Multi‑objective Genetic Algorithm: NSGA‑II”,**
//! *IEEE Transactions on Evolutionary Computation*, 6 (2): 182‑197 (2002).
//!
//! Key ideas implemented here:
//! 1. **Fast non‑dominated sorting** → assigns a Pareto rank *O(N log N + N²)*.
//! 2. **Crowding‑distance preservation** → density estimator used as a
//!    secondary key to maintain diversity without extra parameters.

//! In *moors*, NSGA‑II is wired from reusable operator bricks:
//!
//! * **Selection:** [`RankAndScoringSelection`]
//! * **Survival:**  [`Nsga2RankCrowdingSurvival`] (rank + crowding distance)
//! * **Crossover / Mutation / Sampling:** user‑provided via the builder.
//!
//! The public API exposes only high‑level controls (population size,
//! iteration budget, etc.); internal operator choices can be overridden if
//! you need custom behaviour.

use crate::{selection::moo::RankAndScoringSelection, survival::moo::Nsga2RankCrowdingSurvival};

create_algorithm!(
    /// NSGA‑II algorithm wrapper.
    ///
    /// This struct is a thin façade over [`GeneticAlgorithmMOO`] preset with
    /// the NSGA‑II survival and selection strategy.
    ///
    /// * **Selection:** [`RankAndScoringSelection`]
    /// * **Survival:**  [`Nsga2RankCrowdingSurvival`] (elitist, crowding‑distance)
    ///
    /// Construct it with [`Nsga2Builder`](crate::algorithms::Nsga2Builder) or
    /// directly via [`Nsga2::new`].  After building, call [`run`](GeneticAlgorithmMOO::run)
    /// and then [`population`](GeneticAlgorithmMOO::population) to retrieve the
    /// final non‑dominated set.
    ///
    /// For algorithmic details, see:
    /// K. Deb *et al.* (2002), *IEEE TEC 6 (2)*, 182‑197.
    Nsga2,
    RankAndScoringSelection,
    Nsga2RankCrowdingSurvival
);

impl<S, Cross, Mut, F, G, DC> Default for Nsga2Builder<S, Cross, Mut, F, G, DC>
where
    S: SamplingOperator,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: FitnessFn<Dim = ndarray::Ix2>,
    G: ConstraintsFn,
    DC: PopulationCleaner,
    AlgorithmMOOBuilder<
        S,
        RankAndScoringSelection,
        Nsga2RankCrowdingSurvival,
        Cross,
        Mut,
        F,
        G,
        DC,
    >: Default,
{
    fn default() -> Self {
        let mut inner: AlgorithmMOOBuilder<
            S,
            RankAndScoringSelection,
            Nsga2RankCrowdingSurvival,
            Cross,
            Mut,
            F,
            G,
            DC,
        > = Default::default();
        inner = inner
            .selector(RankAndScoringSelection::default())
            .survivor(Nsga2RankCrowdingSurvival);
        Nsga2Builder {
            inner_builder: inner,
        }
    }
}
