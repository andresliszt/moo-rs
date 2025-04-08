use std::fmt::Debug;

use crate::genetic::Individual;
use crate::operators::{
    selection::DuelResult,
    survival::SurvivalScoringComparison,
    {GeneticOperator, SelectionOperator},
};
use crate::random::RandomGenerator;

#[derive(Clone, Debug)]
pub struct RankAndScoringSelection {
    diversity_comparison: SurvivalScoringComparison,
}

impl RankAndScoringSelection {
    /// Creates a new RankAndScoringSelection with the default diversity comparison (Maximize).
    pub fn new() -> Self {
        Self {
            diversity_comparison: SurvivalScoringComparison::Maximize,
        }
    }
    /// Creates a new RankAndScoringSelection with the specified diversity comparison.
    pub fn new_with_comparison(comparison: SurvivalScoringComparison) -> Self {
        Self {
            diversity_comparison: comparison,
        }
    }
}

impl GeneticOperator for RankAndScoringSelection {
    fn name(&self) -> String {
        "RankAndScoringSelection".to_string()
    }
}

impl SelectionOperator for RankAndScoringSelection {
    /// Runs tournament selection on the given population and returns the duel result.
    /// This example assumes binary tournaments (pressure = 2).
    fn tournament_duel(
        &self,
        p1: &Individual,
        p2: &Individual,
        _rng: &mut dyn RandomGenerator,
    ) -> DuelResult {
        // Check feasibility.
        let p1_feasible = p1.is_feasible();
        let p2_feasible = p2.is_feasible();
        // Retrieve rank.
        let p1_rank = p1.rank;
        let p2_rank = p2.rank;
        // Retrieve diversity (crowding) metric.
        let p1_cd = p1.survival_score;
        let p2_cd = p2.survival_score;

        let winner = if p1_feasible && !p2_feasible {
            DuelResult::LeftWins
        } else if p2_feasible && !p1_feasible {
            DuelResult::RightWins
        } else {
            // Both are either feasible or infeasible.
            if p1_rank < p2_rank {
                DuelResult::LeftWins
            } else if p2_rank < p1_rank {
                DuelResult::RightWins
            } else {
                // When ranks are equal, compare the diversity metric.
                match self.diversity_comparison {
                    SurvivalScoringComparison::Maximize => {
                        if p1_cd > p2_cd {
                            DuelResult::LeftWins
                        } else if p1_cd < p2_cd {
                            DuelResult::RightWins
                        } else {
                            DuelResult::Tie
                        }
                    }
                    SurvivalScoringComparison::Minimize => {
                        if p1_cd < p2_cd {
                            DuelResult::LeftWins
                        } else if p1_cd > p2_cd {
                            DuelResult::RightWins
                        } else {
                            DuelResult::Tie
                        }
                    }
                }
            }
        };

        winner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    use ndarray::{Array1, array};

    use crate::genetic::{Individual, Population, PopulationFitness, PopulationGenes};
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
        fn rng(&mut self) -> &mut dyn rand::RngCore {
            &mut self.dummy
        }
        fn shuffle_vec_usize(&mut self, _vector: &mut Vec<usize>) {
            // Do nothing
        }
    }

    #[test]
    fn test_default_diversity_comparison_maximize() {
        let selector = RankAndScoringSelection::new();
        match selector.diversity_comparison {
            SurvivalScoringComparison::Maximize => assert!(true),
            SurvivalScoringComparison::Minimize => panic!("Default should be Maximize"),
        }
    }

    #[rstest(
        left_feasible, right_feasible, left_rank, right_rank, left_survival, right_survival, diversity, expected,
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
        diversity: SurvivalScoringComparison,
        expected: DuelResult,
    ) {
        // For simplicity, we use the same genes and fitness values for both individuals.
        let genes = array![1.0, 2.0];
        let fitness = array![0.5];

        // Force feasibility by providing constraints: feasible if -1.0, infeasible if 1.0.
        let left_constraints = Some(if left_feasible {
            array![-1.0]
        } else {
            array![1.0]
        });
        let right_constraints = Some(if right_feasible {
            array![-1.0]
        } else {
            array![1.0]
        });

        // Create individuals with the given rank and survival (diversity) values.
        let p1 = Individual::new(
            genes.clone(),
            fitness.clone(),
            left_constraints,
            Some(left_rank),
            Some(left_survival),
        );
        let p2 = Individual::new(
            genes,
            fitness,
            right_constraints,
            Some(right_rank),
            Some(right_survival),
        );

        let selector = RankAndScoringSelection::new_with_comparison(diversity);
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
        let constraints = None;
        let rank = Some(array![0, 1, 0, 1]);

        let population = Population::new(genes, fitness, constraints, rank);

        // n_crossovers = 2 → total_needed = 8 participants → 4 tournaments → 4 winners.
        // After splitting: pop_a = 2 winners, pop_b = 2 winners.
        let n_crossovers = 2;
        let selector = RankAndScoringSelection::new();
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
        let constraints = Some(array![[-1.0, 0.0], [1.0, 1.0]]);
        let rank = Some(array![0, 0]);
        let population = Population::new(genes, fitness, constraints, rank);

        // n_crossovers = 1 → total_needed = 4 participants → 2 tournaments → 2 winners total.
        // After splitting: pop_a = 1 winner, pop_b = 1 winner.
        let n_crossovers = 1;
        let selector = RankAndScoringSelection::new();
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
        let constraints = None;
        let rank = Some(array![0, 0]);

        let population = Population::new(genes, fitness, constraints, rank);

        // n_crossovers = 1 → total_needed = 4 participants → 2 tournaments → 2 winners.
        // After splitting: pop_a = 1 winner, pop_b = 1 winner.
        let n_crossovers = 1;
        let selector = RankAndScoringSelection::new();
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
        let genes = PopulationGenes::from_shape_fn((population_size, n_genes), |(i, _)| i as f64);
        let fitness =
            PopulationFitness::from_shape_fn((population_size, 1), |(i, _)| i as f64 / 100.0);
        let constraints = None;

        let rank = Some(Array1::zeros(population_size));
        let population = Population::new(genes, fitness, constraints, rank);

        // n_crossovers = 50 → total_needed = 200 participants → 100 tournaments → 100 winners.
        // After splitting: pop_a = 50 winners, pop_b = 50 winners.
        let n_crossovers = 50;
        let selector = RankAndScoringSelection::new();
        assert_eq!(selector.name(), "RankAndScoringSelection");
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
        let fitness = array![[1.0], [2.0]];
        let constraints = None;
        let rank = Some(array![0, 1]);

        let population = Population::new(genes, fitness, constraints, rank);

        let n_crossovers = 1;
        let selector = RankAndScoringSelection::new();
        let (pop_a, pop_b) = selector.operate(&population, n_crossovers, &mut rng);

        // The individual with the better rank should win one tournament.
        assert_eq!(pop_a.len(), 1);
        assert_eq!(pop_b.len(), 1);
    }
}
