use ndarray::Array2;

use crate::{
    algorithms::{AlgorithmBuilder, AlgorithmBuilderError, GeneticAlgorithm},
    create_algorithm_and_builder,
    duplicates::PopulationCleaner,
    evaluator::{ConstraintsFn, FitnessFn},
    operators::{
        CrossoverOperator, MutationOperator, SamplingOperator, SelectionOperator, SurvivalOperator,
        selection::moo::RandomSelection,
        survival::moo::{Nsga3ReferencePoints, Nsga3ReferencePointsSurvival},
    },
};

pub type Nsga3<S, Cross, Mut, F, G, DC> =
    GeneticAlgorithm<S, RandomSelection, Nsga3ReferencePointsSurvival, Cross, Mut, F, G, DC>;

pub type Nsga3Builder<S, Cross, Mut, F, G, DC> =
    AlgorithmBuilder<S, RandomSelection, Nsga3ReferencePointsSurvival, Cross, Mut, F, G, DC>;

// ⚠️ impl especializado SOLO para ElitistSurvival
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
    /// Aplana los argumentos de ElitistSurvival y lo construye internamente.
    pub fn survivor(mut self, elite_ratio: f32) -> Self {
        self = self.survivor(value);
        self
    }

    /// También puedes aplanar otros parámetros si tu `ElitistSurvival` los necesita
    pub fn survival_elitist_with(mut self, elite_ratio: f32, min_elite: usize) -> Self {
        self.survival = Some(ElitistSurvival::with_min(elite_ratio, min_elite));
        self
    }
}
