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
