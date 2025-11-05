//! # SPEA‑2 – Strength‑Pareto Evolutionary Algorithm II
//!
//! Implementation of
//! **Eckart Zitzler, Marco Laumanns & Lothar Thiele,
//! “SPEA2: Improving the Strength Pareto Evolutionary Algorithm”,
//! Technical Report 103, Computer Engineering and Networks Laboratory (TIK),
//! ETH Zürich, 2001.**
//!
//! SPEA‑2 maintains two populations (archive + current) and assigns each
//! individual a raw‑fitness value derived from *how many* solutions it dominates
//! (*strength*) and *by how many* it is dominated.  Diversity is preserved with
//! a *k‑nearest‑neighbour* density estimator.
//!
//! In *moors*, SPEA‑2 is wired from reusable operator bricks:
//!
//! * **Selection:** [`RankAndScoringSelection`] (only survival‑score is used)
//! * **Survival:**  [`Spea2KnnSurvival`] (strength + k‑NN density)
//! * **Crossover / Mutation / Sampling:** user‑provided via the builder.
//!
//! The default configuration keeps a secondary **archive** whose size equals
//! the main population; truncation is handled by the k‑NN density measure.
//!

use crate::{
    algorithms::AlgorithmBuilderError,
    create_algorithm_and_builder,
    duplicates::PopulationCleaner,
    evaluator::{ConstraintsFn, FitnessFn},
    operators::{
        CrossoverOperator, MutationOperator, SamplingOperator, SelectionOperator, SurvivalOperator,
        selection::moo::RankAndScoringSelection,
        survival::moo::{Spea2KnnSurvival, SurvivalScoringComparison},
    },
};

create_algorithm_and_builder!(
    /// SPEA-II algorithm wrapper.
    ///
    /// This struct is a thin facade over [`GeneticAlgorithm`] preset with
    /// the SPEA-II survival and selection strategy.
    ///
    /// * **Selection:** [`RankAndScoringSelection`]
    /// * **Survival:**  [`Spea2KnnSurvival`] (elitist, k-nearest neighbors density)
    ///
    /// Construct it with [`Spea2Builder`](crate::algorithms::Spea2Builder).
    /// After building, call [`run`](GeneticAlgorithm::run)
    /// and then [`population`](GeneticAlgorithm::population) to retrieve the
    /// final non-dominated set.
    ///
    /// For algorithmic details, see:
    /// Eckart Zitzler, Marco Laumanns, and Lothar Thiele (2001),
    /// "SPEA2: Improving the Strength Pareto Evolutionary Algorithm",
    /// TIK-Report 103, Computer Engineering and Networks Laboratory,
    /// ETH Zurich, Switzerland, 2001.
    Spea2,
    RankAndScoringSelection,
    Spea2KnnSurvival,
    override_build_method = true
);

impl<S, Cross, Mut, F, G, DC> Spea2Builder<S, Cross, Mut, F, G, DC>
where
    S: SamplingOperator,
    RankAndScoringSelection: SelectionOperator<FDim = F::Dim>,
    Spea2KnnSurvival: SurvivalOperator<FDim = F::Dim>,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: FitnessFn,
    G: ConstraintsFn,
    DC: PopulationCleaner,
{
    pub fn build(mut self) -> Result<Spea2<S, Cross, Mut, F, G, DC>, AlgorithmBuilderError> {
        // Selector operator uses scoring survival given by the raw fitness but it doesn't use rank
        let selector =
            RankAndScoringSelection::new(false, true, SurvivalScoringComparison::Maximize);
        self.inner = self.inner.selector(selector);
        Ok(self.inner.build()?)
    }
}
