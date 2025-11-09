use crate::operators::{
    selection::{SelectionOperator, moo::RankAndScoringSelection},
    survival::moo::SurvivalScoringComparison,
};

#[derive(Debug, Clone)]
pub struct Rnsga2RankScoringSelection(RankAndScoringSelection);

impl Rnsga2RankScoringSelection {
    pub fn new() -> Self {
        Self(RankAndScoringSelection::new(
            true,
            true,
            SurvivalScoringComparison::Minimize,
        ))
    }
}

impl SelectionOperator for Rnsga2RankScoringSelection {
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
