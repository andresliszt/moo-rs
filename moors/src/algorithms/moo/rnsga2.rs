//! # R‑NSGA‑II – Reference‑Point‑Guided NSGA‑II
//!
//! Implementation inspired by
//! **R. Imada, H. Ishibuchi & Y. Nojima,
//! “Reference Point–Based NSGA‑II for Preference‑Based Many‑Objective
//! Optimization”, in *Proc. GECCO 2017*, pp. 1923‑1930.**
//!
//! R‑NSGA‑II biases the classic NSGA‑II ranking/crowding procedure toward
//! **user‑supplied reference points**, enabling preference‑based search without
//! altering the core fast non‑dominated sorting.  At each generation it:
//!
//! 1. Performs NSGA‑II non‑dominated sorting.
//! 2. Computes a *reference‑distance* score (to the closest point in the user
//!    set) **to be *minimized***.
//! 3. Uses that score—rather than crowding distance—as the secondary key
//!    during survival selection.
//!
//! In *moors*, R‑NSGA‑II is wired from reusable operator bricks:
//!
//! * **Selection:** [`RankAndScoringSelection`] (`use_rank = true`, scoring **minimised**)
//! * **Survival:**  [`Rnsga2ReferencePointsSurvival`] (rank + reference‑distance)
//! * **Crossover / Mutation / Sampling:** user‑provided via the builder.
//!
//! Pass your reference point matrix (`Array2<f64>`) and an `epsilon` tolerance
//! to [`Rnsga2::new`]; the survivor will treat individuals whose distance to a
//! point ≤ ε as equally preferred.
//!

use crate::{
    selection::moo::RankAndScoringSelection,
    survival::moo::{Rnsga2ReferencePointsSurvival, SurvivalScoringComparison},
};

create_algorithm!(
    /// R‑NSGA‑II algorithm wrapper.
    ///
    /// Thin façade around [`GeneticAlgorithmMOO`] pre‑configured with
    /// reference‑distance survival and a minimise‑the‑score selector.
    ///
    /// * **Selection:** [`RankAndScoringSelection`] (`SurvivalScoringComparison::Minimize`)
    /// * **Survival:**  [`Rnsga2ReferencePointsSurvival`]
    /// * **Paper:** Imada et al. 2017 (*GECCO*)
    ///
    /// Build via [`Rnsga2Builder`](crate::algorithms::Rnsga2Builder) or directly
    /// with [`Rnsga2::new`]; then call `run()` and `population()` to retrieve the
    /// preference‑biased Pareto set.
    Rnsga2,
    RankAndScoringSelection,
    Rnsga2ReferencePointsSurvival
);

impl<S, Cross, Mut, F, G, DC> Default for Rnsga2Builder<S, Cross, Mut, F, G, DC>
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
        Rnsga2ReferencePointsSurvival,
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
            Rnsga2ReferencePointsSurvival,
            Cross,
            Mut,
            F,
            G,
            DC,
        > = Default::default();

        let selector =
            RankAndScoringSelection::new(true, true, SurvivalScoringComparison::Minimize);

        inner = inner.selector(selector);
        Rnsga2Builder {
            inner_builder: inner,
        }
    }
}
