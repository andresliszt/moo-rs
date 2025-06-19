use crate::operators::{selection::moo::RankAndScoringSelection, survival::moo::AgeMoeaSurvival};

create_algorithm!(
    /// SPEA‑II algorithm wrapper.
    AgeMoea,
    RankAndScoringSelection,
    AgeMoeaSurvival
);

impl<S, Cross, Mut, F, G, DC> Default for AgeMoeaBuilder<S, Cross, Mut, F, G, DC>
where
    S: SamplingOperator,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: FitnessFn<Dim = ndarray::Ix2>,
    G: ConstraintsFn,
    DC: PopulationCleaner,
    AlgorithmMOOBuilder<S, RankAndScoringSelection, AgeMoeaSurvival, Cross, Mut, F, G, DC>: Default,
{
    fn default() -> Self {
        let mut inner: AlgorithmMOOBuilder<
            S,
            RankAndScoringSelection,
            AgeMoeaSurvival,
            Cross,
            Mut,
            F,
            G,
            DC,
        > = Default::default();
        inner = inner
            .selector(RankAndScoringSelection::default())
            .survivor(AgeMoeaSurvival);
        AgeMoeaBuilder {
            inner_builder: inner,
        }
    }
}
