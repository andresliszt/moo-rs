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

use ndarray::Array1;

use crate::{
    algorithms::AlgorithmBuilderError,
    create_algorithm_and_builder,
    duplicates::PopulationCleaner,
    evaluator::{ConstraintsFn, FitnessFn},
    operators::{
        CrossoverOperator, MutationOperator, SamplingOperator, SelectionOperator, SurvivalOperator,
        selection::moo::RankAndScoringSelection,
        survival::moo::{IbeaHyperVolumeSurvivalOperator, SurvivalScoringComparison},
    },
};

create_algorithm_and_builder!(
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
    IbeaHyperVolumeSurvivalOperator,
    extras = [reference: Array1<f64>, kappa: f64],
    override_build_method = true
);

impl<S, Cross, Mut, F, G, DC> IbeaBuilder<S, Cross, Mut, F, G, DC>
where
    S: SamplingOperator,
    RankAndScoringSelection: SelectionOperator<FDim = F::Dim>,
    IbeaHyperVolumeSurvivalOperator: SurvivalOperator<FDim = F::Dim>,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: FitnessFn,
    G: ConstraintsFn,
    DC: PopulationCleaner,
{
    pub fn build(mut self) -> Result<Ibea<S, Cross, Mut, F, G, DC>, AlgorithmBuilderError> {
        let kappa = self.kappa.unwrap_or(0.05);

        let rp = self
            .reference
            .ok_or(AlgorithmBuilderError::UninitializedField(
                "reference_points",
            ))?;

        let survivor = IbeaHyperVolumeSurvivalOperator::new(rp, kappa);
        // Selector operator uses scoring survival given by the raw fitness but it doesn't use rank
        let selector =
            RankAndScoringSelection::new(false, true, SurvivalScoringComparison::Maximize);
        self.inner = self.inner.survivor(survivor);
        self.inner = self.inner.selector(selector);
        Ok(self.inner.build()?)
    }
}
