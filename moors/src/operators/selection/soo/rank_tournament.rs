use crate::genetic::{D01, IndividualSOO};
use crate::operators::selection::{DuelResult, SelectionOperator};
use crate::random::RandomGenerator;

#[derive(Debug, Clone)]
pub struct RankSelection;

impl SelectionOperator for RankSelection {
    type FDim = ndarray::Ix1;
    /// Runs tournament selection on the given population and returns the duel result.
    /// This assumes binary tournaments (pressure = 2).
    fn tournament_duel<'a, ConstrDim>(
        &self,
        p1: &IndividualSOO<'a, ConstrDim>,
        p2: &IndividualSOO<'a, ConstrDim>,
        _rng: &mut impl RandomGenerator,
    ) -> DuelResult
    where
        ConstrDim: D01,
    {
        // Feasibility dominates everything
        if let result @ DuelResult::LeftWins | result @ DuelResult::RightWins =
            Self::feasibility_dominates(p1, p2)
        {
            return result;
        }

        match p1.rank.cmp(&p2.rank) {
            std::cmp::Ordering::Less => DuelResult::LeftWins,
            std::cmp::Ordering::Greater => DuelResult::RightWins,
            std::cmp::Ordering::Equal => DuelResult::Tie,
        }
    }
}
