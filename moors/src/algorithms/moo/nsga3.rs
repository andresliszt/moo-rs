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

use ndarray::Array2;

use crate::{
    algorithms::AlgorithmBuilderError,
    create_algorithm_and_builder,
    duplicates::PopulationCleaner,
    evaluator::{ConstraintsFn, FitnessFn},
    operators::{
        CrossoverOperator, MutationOperator, SamplingOperator, SelectionOperator, SurvivalOperator,
        selection::moo::RandomSelection,
        survival::moo::{Nsga3ReferencePoints, Nsga3ReferencePointsSurvival},
    },
};

create_algorithm_and_builder!(
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
    RandomSelection,
    Nsga3ReferencePointsSurvival,
    extras = [ reference_points: Array2<f64>, are_aspirational: bool ],
    override_build_method = true
);

impl<S, Cross, Mut, F, G, DC> Nsga3Builder<S, Cross, Mut, F, G, DC>
where
    S: SamplingOperator,
    RandomSelection: SelectionOperator<FDim = F::Dim>,
    Nsga3ReferencePointsSurvival: SurvivalOperator<FDim = F::Dim>,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: FitnessFn,
    G: ConstraintsFn,
    DC: PopulationCleaner,
{
    pub fn build(mut self) -> Result<Nsga3<S, Cross, Mut, F, G, DC>, AlgorithmBuilderError> {
        let aspirational = self.are_aspirational.unwrap_or(false);
        let rp = self
            .reference_points
            .ok_or(AlgorithmBuilderError::UninitializedField(
                "reference_points",
            ))?;

        let rp = Nsga3ReferencePoints::new(rp, aspirational);
        let survivor = Nsga3ReferencePointsSurvival::new(rp);
        self.inner = self.inner.survivor(survivor);
        Ok(self.inner.build()?)
    }
}
