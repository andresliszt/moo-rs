use std::fmt::Debug;

use crate::genetic::Individual;
use crate::operators::selection::{DuelResult, GeneticOperator, SelectionOperator};
use crate::random::RandomGenerator;

#[derive(Clone, Debug)]
pub struct RandomSelection {}

impl RandomSelection {
    pub fn new() -> Self {
        Self {}
    }
}

impl GeneticOperator for RandomSelection {
    fn name(&self) -> String {
        "RandomSelection".to_string()
    }
}

impl SelectionOperator for RandomSelection {
    fn tournament_duel(
        &self,
        p1: &Individual,
        p2: &Individual,
        rng: &mut dyn RandomGenerator,
    ) -> DuelResult {
        let p1_feasible = p1.is_feasible();
        let p2_feasible = p2.is_feasible();

        // If exactly one is feasible, that one automatically wins:
        if p1_feasible && !p2_feasible {
            DuelResult::LeftWins
        } else if p2_feasible && !p1_feasible {
            DuelResult::RightWins
        } else {
            // Otherwise, both are feasible or both are infeasible => random winner.
            if rng.gen_bool(0.5) {
                DuelResult::LeftWins
            } else {
                DuelResult::RightWins
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{RandomGenerator, TestDummyRng};
    use ndarray::array;
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
        fn rng(&mut self) -> &mut dyn rand::RngCore {
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
    fn test_tournament_duel_parametrized_rstest(
        p1_constraint: f64,
        p2_constraint: f64,
        rng_value: bool,
        expected: DuelResult,
    ) {
        // Define genes and fitness as arrays of f64.
        let genes = array![1.0, 2.0, 3.0];
        let fitness = array![0.5];

        let p1 = Individual::new(
            genes.clone(),
            fitness.clone(),
            Some(array![p1_constraint]),
            None,
            None,
        );
        let p2 = Individual::new(genes, fitness, Some(array![p2_constraint]), None, None);

        let selector = RandomSelection::new();
        assert_eq!(selector.name(), "RandomSelection");
        let mut rng = FakeRandomGenerator::new(rng_value);
        let result = selector.tournament_duel(&p1, &p2, &mut rng);
        assert_eq!(result, expected);
    }
}
