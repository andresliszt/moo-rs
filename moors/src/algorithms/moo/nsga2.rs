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

create_algorithm_and_builder!(Nsga2, RankAndScoringSelection, Nsga2RankCrowdingSurvival);
