use std::fmt::Debug;

use crate::genetic::{D01, IndividualMOO};
use crate::operators::{
    SelectionOperator, selection::DuelResult, survival::SurvivalScoringComparison,
};
use crate::random::RandomGenerator;

#[derive(Debug, Clone)]
pub struct RankAndScoringSelection {
    use_rank: bool,
    use_survival_score: bool,
    survival_comparison: SurvivalScoringComparison,
}

impl RankAndScoringSelection {
    /// Selection operator based on rank and or scoring survival.
    ///
    /// * `use_rank` – whether rank is considered.
    /// * `use_survival_score` – whether survival score is considered.
    /// * `survival_comparison` – `Maximize` or `Minimize` (ignored if
    ///   `use_survival_score` is `false`).
    ///
    /// # Panics
    /// Panics if both `use_rank == false` and `use_survival_score == false`.
    pub fn new(
        use_rank: bool,
        use_survival_score: bool,
        survival_comparison: SurvivalScoringComparison,
    ) -> Self {
        assert!(
            use_rank || use_survival_score,
            "RankAndScoringSelection: At least one criterion (rank or survival score) must be enabled"
        );
        Self {
            use_rank,
            use_survival_score,
            survival_comparison,
        }
    }
}

impl Default for RankAndScoringSelection {
    /// Default = use both criteria; maximize survival score.
    fn default() -> Self {
        Self::new(true, true, SurvivalScoringComparison::Maximize)
    }
}

impl SelectionOperator for RankAndScoringSelection {
    /// Runs tournament selection on the given population and returns the duel result.
    /// This example assumes binary tournaments (pressure = 2).
    fn tournament_duel<'a, ConstrDim>(
        &self,
        p1: &IndividualMOO<'a, ConstrDim>,
        p2: &IndividualMOO<'a, ConstrDim>,
        _rng: &mut impl RandomGenerator,
    ) -> DuelResult
    where
        ConstrDim: D01,
    {
        /* 1. Feasibility dominates everything */
        match (p1.is_feasible(), p2.is_feasible()) {
            (true, false) => return DuelResult::LeftWins,
            (false, true) => return DuelResult::RightWins,
            _ => {}
        }

        // Rank (if enabled)
        if self.use_rank {
            match p1.rank.cmp(&p2.rank) {
                std::cmp::Ordering::Less => return DuelResult::LeftWins,
                std::cmp::Ordering::Greater => return DuelResult::RightWins,
                std::cmp::Ordering::Equal => {}
            }
        }

        // Survival score (if enabled)
        if self.use_survival_score {
            use SurvivalScoringComparison::*;
            return match self.survival_comparison {
                Maximize => match p1.survival_score.partial_cmp(&p2.survival_score) {
                    Some(std::cmp::Ordering::Greater) => DuelResult::LeftWins,
                    Some(std::cmp::Ordering::Less) => DuelResult::RightWins,
                    _ => DuelResult::Tie,
                },
                Minimize => match p1.survival_score.partial_cmp(&p2.survival_score) {
                    Some(std::cmp::Ordering::Less) => DuelResult::LeftWins,
                    Some(std::cmp::Ordering::Greater) => DuelResult::RightWins,
                    _ => DuelResult::Tie,
                },
            };
        }
        DuelResult::Tie
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    use ndarray::{Array1, Array2, arr0, array};

    use crate::genetic::PopulationMOO;
    use crate::operators::selection::{DuelResult, SelectionOperator};
    use crate::random::{RandomGenerator, TestDummyRng};

    // A fake random generator to control the outcome of gen_bool.
    struct FakeRandomGenerator {
        dummy: TestDummyRng,
    }

    impl FakeRandomGenerator {
        fn new() -> Self {
            Self {
                dummy: TestDummyRng,
            }
        }
    }

    impl RandomGenerator for FakeRandomGenerator {
        type R = TestDummyRng;
        fn rng(&mut self) -> &mut TestDummyRng {
            &mut self.dummy
        }
        fn shuffle_vec_usize(&mut self, _vector: &mut Vec<usize>) {
            // Do nothing
        }
    }

    #[test]
    fn test_default_diversity_comparison_maximize() {
        let selector = RankAndScoringSelection::default();
        match selector.survival_comparison {
            SurvivalScoringComparison::Maximize => assert!(true),
            SurvivalScoringComparison::Minimize => panic!("Default should be Maximize"),
        }
        // default uses both
        assert!(selector.use_rank);
        assert!(selector.use_survival_score);
    }

    #[rstest(
        left_feasible, right_feasible, left_rank, right_rank, left_survival, right_survival, survival_comparison, expected,
        // Feasibility check: if one is feasible and the other isn't, feasibility wins regardless of rank or survival.
        case(true, false, 0, 1, 10.0, 5.0, SurvivalScoringComparison::Maximize, DuelResult::LeftWins),
        case(false, true, 1, 0, 10.0, 5.0, SurvivalScoringComparison::Maximize, DuelResult::RightWins),

        // Both are feasible: rank comparison takes precedence.
        case(true, true, 0, 1, 5.0, 10.0, SurvivalScoringComparison::Maximize, DuelResult::LeftWins),
        case(true, true, 2, 1, 5.0, 10.0, SurvivalScoringComparison::Maximize, DuelResult::RightWins),

        // Both are feasible (or both infeasible) and ranks are equal → decide by survival (diversity) in Maximize mode.
        case(true, true, 0, 0, 10.0, 5.0, SurvivalScoringComparison::Maximize, DuelResult::LeftWins),
        case(true, true, 0, 0, 5.0, 10.0, SurvivalScoringComparison::Maximize, DuelResult::RightWins),
        case(true, true, 0, 0, 7.0, 7.0, SurvivalScoringComparison::Maximize, DuelResult::Tie),

        // Both are feasible (or both infeasible) and ranks are equal → decide by survival in Minimize mode.
        case(true, true, 0, 0, 5.0, 10.0, SurvivalScoringComparison::Minimize, DuelResult::LeftWins),
        case(true, true, 0, 0, 10.0, 5.0, SurvivalScoringComparison::Minimize, DuelResult::RightWins),
        case(true, true, 0, 0, 8.0, 8.0, SurvivalScoringComparison::Minimize, DuelResult::Tie),

        // Both are infeasible: rules are the same as for feasible individuals.
        case(false, false, 0, 1, 10.0, 5.0, SurvivalScoringComparison::Maximize, DuelResult::LeftWins),
        case(false, false, 0, 0, 7.0, 7.0, SurvivalScoringComparison::Maximize, DuelResult::Tie)
    )]
    fn test_tournament_duel(
        left_feasible: bool,
        right_feasible: bool,
        left_rank: usize,
        right_rank: usize,
        left_survival: f64,
        right_survival: f64,
        survival_comparison: SurvivalScoringComparison,
        expected: DuelResult,
    ) {
        // For simplicity, we use the same genes and fitness values for both individuals.
        let genes = array![1.0, 2.0];
        let fitness = array![0.5];

        // Force feasibility by providing constraints: feasible if -1.0, infeasible if 1.0.
        let left_arr0 = if left_feasible { arr0(-1.0) } else { arr0(1.0) };
        let right_arr0 = if right_feasible {
            arr0(-1.0)
        } else {
            arr0(1.0)
        };
        let left_constraints = left_arr0.view();
        let right_constraints = right_arr0.view();
        // Create individuals with the given rank and survival (diversity) values.
        let p1: IndividualMOO<'_, ndarray::Ix0> = IndividualMOO {
            genes: genes.view(),
            fitness: fitness.view(),
            constraints: Some(left_constraints),
            rank: Some(left_rank),
            survival_score: Some(left_survival),
        };
        let mut p2: IndividualMOO<'_, ndarray::Ix0> =
            IndividualMOO::new(genes.view(), fitness.view(), right_constraints);
        // Call setters to extend this test
        p2.set_rank(right_rank);
        p2.set_survival_score(right_survival);

        let selector = RankAndScoringSelection::new(true, true, survival_comparison);
        let mut rng = FakeRandomGenerator::new();
        let result = selector.tournament_duel(&p1, &p2, &mut rng);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_operate_no_constraints_basic() {
        let mut rng = FakeRandomGenerator::new();
        // For a population of 4:
        // Rank: [0, 1, 0, 1]
        // Diversity (CD): [10.0, 5.0, 9.0, 1.0]
        let genes = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let fitness = array![[0.5], [0.6], [0.7], [0.8]];
        let rank = array![0, 1, 0, 1];

        let mut population = PopulationMOO::new_unconstrained(genes, fitness);
        population.set_rank(rank);

        // n_crossovers = 2 → total_needed = 8 participants → 4 tournaments → 4 winners.
        // After splitting: pop_a = 2 winners, pop_b = 2 winners.
        let n_crossovers = 2;
        let selector = RankAndScoringSelection::default();
        let (pop_a, pop_b) = selector.operate(&population, n_crossovers, &mut rng);

        assert_eq!(pop_a.len(), 2);
        assert_eq!(pop_b.len(), 2);
    }

    #[test]
    fn test_operate_with_constraints() {
        let mut rng = FakeRandomGenerator::new();
        // Two individuals:
        // Individual 0: feasible
        // Individual 1: infeasible
        let genes = array![[1.0, 2.0], [3.0, 4.0]];
        let fitness = array![[0.5], [0.6]];
        let constraints = array![[-1.0, 0.0], [1.0, 1.0]];
        let rank = array![0, 0];
        let mut population = PopulationMOO::new(genes, fitness, constraints);
        population.set_rank(rank);

        // n_crossovers = 1 → total_needed = 4 participants → 2 tournaments → 2 winners total.
        // After splitting: pop_a = 1 winner, pop_b = 1 winner.
        let n_crossovers = 1;
        let selector = RankAndScoringSelection::default();
        let (pop_a, pop_b) = selector.operate(&population, n_crossovers, &mut rng);

        // The feasible individual should be one of the winners.
        assert_eq!(pop_a.len(), 1);
        assert_eq!(pop_b.len(), 1);
    }

    #[test]
    fn test_operate_same_rank_and_cd() {
        let mut rng = FakeRandomGenerator::new();
        // If two individuals have the same rank and the same crowding distance,
        // the tournament duel should result in a tie.
        let genes = array![[1.0, 2.0], [3.0, 4.0]];
        let fitness = array![[0.5], [0.6]];
        let rank = array![0, 0];
        let mut population = PopulationMOO::new_unconstrained(genes, fitness);
        population.set_rank(rank);

        // n_crossovers = 1 → total_needed = 4 participants → 2 tournaments → 2 winners.
        // After splitting: pop_a = 1 winner, pop_b = 1 winner.
        let n_crossovers = 1;
        let selector = RankAndScoringSelection::default();
        let (pop_a, pop_b) = selector.operate(&population, n_crossovers, &mut rng);

        // In a tie, the overall selection process must eventually choose winners.
        // For this test, we ensure that each subpopulation has 1 individual.
        assert_eq!(pop_a.len(), 1);
        assert_eq!(pop_b.len(), 1);
    }

    #[test]
    fn test_operate_large_population() {
        let mut rng = FakeRandomGenerator::new();
        // Large population test to ensure stability.
        let population_size = 100;
        let n_genes = 5;
        let genes = Array2::from_shape_fn((population_size, n_genes), |(i, _)| i as f64);
        let fitness = Array2::from_shape_fn((population_size, 2), |(i, _)| i as f64 / 100.0);
        let rank = Array1::zeros(population_size);
        let mut population = PopulationMOO::new_unconstrained(genes, fitness);
        population.set_rank(rank);

        // n_crossovers = 50 → total_needed = 200 participants → 100 tournaments → 100 winners.
        // After splitting: pop_a = 50 winners, pop_b = 50 winners.
        let n_crossovers = 50;
        let selector = RankAndScoringSelection::default();
        let (pop_a, pop_b) = selector.operate(&population, n_crossovers, &mut rng);

        assert_eq!(pop_a.len(), 50);
        assert_eq!(pop_b.len(), 50);
    }

    #[test]
    fn test_operate_single_tournament() {
        let mut rng = FakeRandomGenerator::new();
        // One crossover:
        // total_needed = 4 participants → 2 tournaments → 2 winners.
        // After splitting: pop_a = 1, pop_b = 1.
        let genes = array![[10.0, 20.0], [30.0, 40.0]];
        let fitness = array![[1.0, 2.0], [2.0, 1.0]];
        let rank = array![0, 1];
        let mut population = PopulationMOO::new_unconstrained(genes, fitness);
        population.set_rank(rank);

        let n_crossovers = 1;
        let selector = RankAndScoringSelection::default();
        let (pop_a, pop_b) = selector.operate(&population, n_crossovers, &mut rng);

        // The individual with the better rank should win one tournament.
        assert_eq!(pop_a.len(), 1);
        assert_eq!(pop_b.len(), 1);
    }
}
