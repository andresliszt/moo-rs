use crate::genetic::{D01, IndividualSOO};
use crate::operators::selection::{DuelResult, SelectionOperator};
use crate::random::RandomGenerator;

#[derive(Debug, Clone)]
pub struct RandomSelection {}

impl RandomSelection {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for RandomSelection {
    fn default() -> Self {
        Self::new()
    }
}

impl SelectionOperator for RandomSelection {
    type FDim = ndarray::Ix1;

    fn tournament_duel<'a, ConstrDim>(
        &self,
        p1: &IndividualSOO<'a, ConstrDim>,
        p2: &IndividualSOO<'a, ConstrDim>,
        rng: &mut impl RandomGenerator,
    ) -> DuelResult
    where
        ConstrDim: D01,
    {
        if let result @ DuelResult::LeftWins | result @ DuelResult::RightWins =
            Self::feasibility_dominates(p1, p2)
        {
            return result;
        }
        // Otherwise, both are feasible or both are infeasible => random winner.
        if rng.gen_bool(0.5) {
            DuelResult::LeftWins
        } else {
            DuelResult::RightWins
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{RandomGenerator, TestDummyRng};
    use ndarray::{arr0, array};
    use rstest::rstest;

    // A fake random generator to control the outcome of gen_bool.
    struct FakeRandomGenerator {
        dummy: TestDummyRng,
        value: bool,
    }

    impl FakeRandomGenerator {
        fn new(value: bool) -> Self {
            Self {
                dummy: TestDummyRng,
                value,
            }
        }
    }

    impl RandomGenerator for FakeRandomGenerator {
        type R = TestDummyRng;
        fn rng(&mut self) -> &mut TestDummyRng {
            &mut self.dummy
        }
        fn gen_bool(&mut self, _p: f64) -> bool {
            self.value
        }
    }

    // Parameterized test using rstest
    #[rstest(
        p1_constraint, p2_constraint, rng_value, expected,
        case(0.0, 1.0, true, DuelResult::LeftWins),  // p1 feasible, p2 infeasible
        case(1.0, 0.0, true, DuelResult::RightWins), // p1 infeasible, p2 feasible
        case(0.0, 0.0, true, DuelResult::LeftWins),  // both feasible, RNG true => left wins
        case(0.0, 0.0, false, DuelResult::RightWins),// both feasible, RNG false => right wins
        case(1.0, 1.0, true, DuelResult::LeftWins),  // both infeasible, RNG true => left wins
        case(1.0, 1.0, false, DuelResult::RightWins) // both infeasible, RNG false => right wins
    )]
    fn test_tournament_duel(
        p1_constraint: f64,
        p2_constraint: f64,
        rng_value: bool,
        expected: DuelResult,
    ) {
        // Define genes and fitness as arrays of f64.
        let genes = array![1.0, 2.0, 3.0];
        let fitness = arr0(0.5);
        let p1_constraint_arr0 = arr0(p1_constraint);
        let p2_constraint_arr0 = arr0(p2_constraint);

        let p1: IndividualSOO<ndarray::Ix0> =
            IndividualSOO::new(genes.view(), fitness.view(), p1_constraint_arr0.view());
        let p2 = IndividualSOO::new(genes.view(), fitness.view(), p2_constraint_arr0.view());

        let selector = RandomSelection::new();
        let mut rng = FakeRandomGenerator::new(rng_value);
        let result = selector.tournament_duel(&p1, &p2, &mut rng);
        assert_eq!(result, expected);
    }
}
