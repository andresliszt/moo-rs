use crate::operators::{
    selection::moo::RankAndScoringSelection, survival::moo::SurvivalScoringComparison,
};

#[derive(Debug, Clone)]
pub struct Nsga2RankAndScoringSelection(RankAndScoringSelection);

impl Nsga2RankAndScoringSelection {
    pub fn new() -> Self {
        Self(RankAndScoringSelection::new(
            true,
            true,
            SurvivalScoringComparison::Maximize,
        ))
    }
}

impl crate::operators::selection::SelectionOperator for Nsga2RankAndScoringSelection {
    type FDim = <RankAndScoringSelection as crate::operators::selection::SelectionOperator>::FDim;

    fn tournament_duel<'a, ConstrDim>(
        &self,
        p1: &crate::genetic::IndividualMOO<'a, ConstrDim>,
        p2: &crate::genetic::IndividualMOO<'a, ConstrDim>,
        rng: &mut impl crate::random::RandomGenerator,
    ) -> crate::operators::selection::DuelResult
    where
        ConstrDim: crate::genetic::D01,
    {
        self.0.tournament_duel(p1, p2, rng)
    }
}
