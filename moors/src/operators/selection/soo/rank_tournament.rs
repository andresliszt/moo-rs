use crate::genetic::{D01, IndividualSOO};
use crate::operators::selection::{DuelResult, SelectionOperator};
use crate::random::RandomGenerator;

#[derive(Debug, Clone)]
pub struct RankSelection;

impl RankSelection {
    pub fn new() -> Self {
        Self {}
    }
}

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
        match (p1.is_feasible(), p2.is_feasible()) {
            (true, false) => return DuelResult::LeftWins,
            (false, true) => return DuelResult::RightWins,
            _ => {}
        }

        match p1.rank.cmp(&p2.rank) {
            std::cmp::Ordering::Less => return DuelResult::LeftWins,
            std::cmp::Ordering::Greater => return DuelResult::RightWins,
            std::cmp::Ordering::Equal => return DuelResult::Tie,
        }
    }
}
