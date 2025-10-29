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
    Nsga3,
    RandomSelection,
    Nsga3ReferencePointsSurvival,
    extras = [ reference_points: Array2<f64>, are_aspirational: bool ],
    override_default_method = true
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
    pub fn build(
        mut self,
    ) -> Result<Nsga3<S, Cross, Mut, F, G, DC>, crate::algorithms::AlgorithmBuilderError> {
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
