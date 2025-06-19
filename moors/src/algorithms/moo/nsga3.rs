//! # NSGA‑III – Reference‑Point‑Based Many‑Objective GA
//!
//! Implementation of
//! **K. Deb & H. Jain,
//! “An Evolutionary Many‑Objective Optimization Algorithm Using
//! Reference‑Point‑Based Nondominated Sorting Approach, Part I:
//! Solving Problems with Box Constraints”,
//! IEEE Transactions on Evolutionary Computation 18 (4): 577‑601 (2014).**
//!
//! NSGA‑III extends NSGA‑II to many‑objective problems (≥ 3 objectives) by
//! replacing crowding‑distance with a *reference‑point association* that drives
//! the population towards a well‑spread Pareto front.
//!
//! In *moors*, NSGA‑III is wired from reusable operator bricks:
//!
//! * **Selection:** [`RandomSelection`] (uniform binary tournament)
//! * **Survival:**  [`Nsga3ReferencePointsSurvival`] (rank + reference‑point niching)
//! * **Crossover / Mutation / Sampling:** user‑provided via the builder.
//!
//! You supply the *reference points*—typically generated with
//! [`Nsga3ReferencePoints::from_simplex_lattice`] or a custom constructor—and
//! the algorithm handles association and niche preservation automatically.

use crate::{selection::moo::RandomSelection, survival::moo::Nsga3ReferencePointsSurvival};

create_algorithm!(
    /// NSGA‑III algorithm wrapper.
    Nsga3,
    RandomSelection,
    Nsga3ReferencePointsSurvival
);

impl<S, Cross, Mut, F, G, DC> Default for Nsga3Builder<S, Cross, Mut, F, G, DC>
where
    S: SamplingOperator,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: FitnessFn<Dim = ndarray::Ix2>,
    G: ConstraintsFn,
    DC: PopulationCleaner,
    AlgorithmMOOBuilder<S, RandomSelection, Nsga3ReferencePointsSurvival, Cross, Mut, F, G, DC>:
        Default,
{
    fn default() -> Self {
        let mut inner: AlgorithmMOOBuilder<
            S,
            RandomSelection,
            Nsga3ReferencePointsSurvival,
            Cross,
            Mut,
            F,
            G,
            DC,
        > = Default::default();
        inner = inner.selector(RandomSelection);
        Nsga3Builder {
            inner_builder: inner,
        }
    }
}
