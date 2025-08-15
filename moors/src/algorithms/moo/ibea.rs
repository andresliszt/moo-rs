//! # IBEA – Indicator-Based Evolutionary Algorithm
//!
//! Implementation of
//! **Eckart Zitzler & Simon Künzli,
//! “Indicator-Based Evolutionary Algorithm for Multiobjective Optimization,”
//! EMO 2004, LNCS 3248, Springer.**
//!
//! IBEA assigns fitness directly from a pairwise **quality indicator** (e.g., hypervolume, ε-indicator).
//! Given κ>0, it builds a matrix `M[i,j] = -exp(-I(i,j)/κ)` (with `M[i,i]=0`) and sets
//! the fitness as `F[j] = Σ_i M[i,j]`. Environmental selection iteratively removes the individual
//! with the **smallest** `F` and updates the remaining fitness by subtracting the removed row of `M`.
//!
//! In *moors*, IBEA is composed from reusable operators:
//! * **Selection:** [`RankAndScoringSelection`] (uses only survival score)
//! * **Survival:**  [`IbeaHyperVolumeSurvivalOperator`] (indicator-driven; hypervolume singleton by default)
//! * **Crossover / Mutation / Sampling:** user-provided via the builder.
//!
//! The default configuration keeps a single population (no external archive).

use crate::{
    create_algorithm,
    selection::moo::RankAndScoringSelection,
    survival::moo::{IbeaHyperVolumeSurvivalOperator, SurvivalScoringComparison},
};

create_algorithm!(
    /// IBEA algorithm wrapper.
    ///
    /// Thin façade over [`GeneticAlgorithm`] preset with IBEA’s selection/survival strategy.
    ///
    /// * **Selection:** [`RankAndScoringSelection`]
    /// * **Survival:**  [`IbeaHyperVolumeSurvivalOperator`] (indicator-driven, κ-controlled)
    ///
    /// Build with [`IbeaBuilder`](crate::algorithms::IbeaBuilder), run with
    /// [`run`](GeneticAlgorithm::run) and retrieve the final population with
    /// [`population`](GeneticAlgorithm::population).
    ///
    /// Reference:
    /// Zitzler & Künzli (2004), *Indicator-Based Evolutionary Algorithm for Multiobjective Optimization*,
    /// EMO 2004, LNCS 3248, Springer.
    Ibea,
    RankAndScoringSelection,
    IbeaHyperVolumeSurvivalOperator
);

impl<S, Cross, Mut, F, G, DC> Default for IbeaBuilder<S, Cross, Mut, F, G, DC>
where
    S: SamplingOperator,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: FitnessFn<Dim = ndarray::Ix2>,
    G: ConstraintsFn,
    DC: PopulationCleaner,
    AlgorithmBuilder<
        S,
        RankAndScoringSelection,
        IbeaHyperVolumeSurvivalOperator,
        Cross,
        Mut,
        F,
        G,
        DC,
    >: Default,
{
    fn default() -> Self {
        let mut inner: AlgorithmBuilder<
            S,
            RankAndScoringSelection,
            IbeaHyperVolumeSurvivalOperator,
            Cross,
            Mut,
            F,
            G,
            DC,
        > = Default::default();

        // Selector operator uses scoring survival given by the raw fitness but it doesn't use rank
        let selector =
            RankAndScoringSelection::new(false, true, SurvivalScoringComparison::Maximize);

        inner = inner.selector(selector);
        IbeaBuilder {
            inner_builder: inner,
        }
    }
}
