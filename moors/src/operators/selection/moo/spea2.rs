use crate::operators::{
    selection::{SelectionOperator, moo::RankAndScoringSelection},
    survival::moo::SurvivalScoringComparison,
};

#[derive(Debug, Clone)]
pub struct Spea2ScoringSelection(RankAndScoringSelection);

impl Spea2ScoringSelection {
    pub fn new() -> Self {
        Self(RankAndScoringSelection::new(
            false,
            true,
            SurvivalScoringComparison::Maximize,
        ))
    }
}

impl SelectionOperator for Spea2ScoringSelection {
    type FDim = <RankAndScoringSelection as SelectionOperator>::FDim;

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
