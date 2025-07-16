use std::cmp::Ordering;

use ndarray::Array1;

use crate::{
    genetic::{D12, PopulationSOO},
    operators::survival::SurvivalOperator,
    random::RandomGenerator,
};

/// A survival operator that selects individuals based on a **penalized fitness** score,
/// which combines the objective value (fitness) and the total constraint violation.
///
/// The selection score for each individual is computed as:
///
/// ```text
/// penalized_score = fitness + constraints_penalty × constraint_violation
/// ```
///
/// Individuals with lower penalized scores are preferred (i.e., minimization).
///
/// This operator allows you to balance the trade-off between feasibility and objective
/// quality:
///
/// - When `constraints_penalty` is small, fitness dominates.
/// - When `constraints_penalty` is large, constraint satisfaction dominates.
///
/// If no constraint violations are present in the population, selection defaults
/// to pure fitness-based minimization.
pub struct FitnessConstraintsPenaltySurvival {
    constraints_penalty: f64,
}

impl FitnessConstraintsPenaltySurvival {
    pub fn new(constraints_penalty: f64) -> Self {
        Self {
            constraints_penalty,
        }
    }
}

impl SurvivalOperator for FitnessConstraintsPenaltySurvival {
    type FDim = ndarray::Ix1;

    fn operate<ConstrDim>(
        &mut self,
        population: PopulationSOO<ConstrDim>,
        num_survive: usize,
        _rng: &mut impl RandomGenerator,
    ) -> PopulationSOO<ConstrDim>
    where
        ConstrDim: D12,
    {
        let pop_size = population.fitness.len();
        let mut indices: Vec<usize> = (0..pop_size).collect();

        if let Some(violations) = &population.constraint_violation_totals {
            let penalty_scores: Vec<f64> = (0..pop_size)
                .map(|i| self.constraints_penalty * violations[i] + population.fitness[i])
                .collect();

            indices.sort_by(|&i, &j| {
                penalty_scores[i]
                    .partial_cmp(&penalty_scores[j])
                    .unwrap_or(Ordering::Equal)
            });
        } else {
            indices.sort_by(|&i, &j| {
                population.fitness[i]
                    .partial_cmp(&population.fitness[j])
                    .unwrap_or(Ordering::Equal)
            });
        }

        let survive_count = num_survive.min(pop_size);
        let selected_indices = &indices[..survive_count];
        let mut selected_population = population.selected(selected_indices);

        selected_population.set_rank(Array1::from_iter(0..survive_count));
        selected_population
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::random::TestDummyRng;
    use ndarray::{Array2, array};

    // A fake random generator; FitnessSurvival does not actually use RNG here,
    // but we need to satisfy the trait bound.
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
            // no-op
        }
    }

    #[test]
    fn selects_lowest_two_fitness() {
        // Four individuals with fitness [0.8, 0.2, 0.5, 0.1].
        // Selecting 2 should keep indices 3 (0.1) and 1 (0.2).
        let genes = Array2::zeros((4, 1));
        let fitness = array![0.8, 0.2, 0.5, 0.1];
        let pop = PopulationSOO::new_unconstrained(genes, fitness);
        let mut rng = FakeRandomGenerator::new();
        let mut selector = FitnessConstraintsPenaltySurvival::new(1.0);
        let survived = selector.operate(pop, 2, &mut rng);

        // survive_count = 2
        assert_eq!(survived.fitness.len(), 2);

        // Survived fitnesses should be [0.1, 0.2]
        let expected_fitness = array![0.1, 0.2];
        assert_eq!(survived.fitness, expected_fitness);

        // Expected ranks: [0, 1]
        let expected_ranks = array![0, 1];
        assert_eq!(survived.rank.unwrap(), expected_ranks);
    }

    #[test]
    fn selects_lowest_two_constraints_violation() {
        // Four individuals with fitness [0.8, 0.2, 0.5, 0.1]
        // With constraints violations 0.0 for first two individuals
        // Selecting 2 should keep these individuals 0, 1 but individual with
        // index 1 is the the one with rank = 0 because it has lowest lower fitness value
        let genes = Array2::zeros((4, 1));
        let fitness = array![0.8, 0.2, 0.5, 0.1];
        let constraints = array![0.0, 0.0, 10.0, 100.0];
        let pop = PopulationSOO::new(genes, fitness, constraints);
        let mut rng = FakeRandomGenerator::new();
        let mut selector = FitnessConstraintsPenaltySurvival::new(1.0);
        let survived = selector.operate(pop, 2, &mut rng);
        // survive_count = 2
        assert_eq!(survived.fitness.len(), 2);
        // Survived fitnesses should be [0.2, 0.8]
        let expected_fitness = array![0.2, 0.8];
        assert_eq!(survived.fitness, expected_fitness);
        // Expected best
        let best = survived.best();
        assert_eq!(best.fitness, array![0.2]);
        assert_eq!(best.constraint_violation_totals.unwrap(), array![0.0]);
    }

    #[test]
    fn constraints_penalty_affects_selection_order() {
        // Two infeasible individuals:
        // - Individual 0: great fitness, high violation
        // - Individual 1: poor fitness, low violation
        let genes = Array2::zeros((2, 1));
        let fitness = array![0.1, 0.9];
        let constraints = array![10.0, 1.0];

        let pop = PopulationSOO::new(genes, fitness, constraints);
        let mut rng = FakeRandomGenerator::new();
        // --- Case 1: Low penalty → fitness dominates
        // Penalized:
        //   Individual 0: 0.1 + 0.001 * 10 = 0.11
        //   Individual 1: 0.9 + 0.001 * 1  = 0.901
        // → Individual 0 should win
        let mut selector_low = FitnessConstraintsPenaltySurvival::new(0.001);
        let survived_low = selector_low.operate(pop.clone(), 1, &mut rng);
        assert_eq!(survived_low.fitness, array![0.1]);

        // --- Case 2: High penalty → constraint violation dominates
        // Penalized:
        //   Individual 0: 0.1 + 1.0 * 10 = 10.1
        //   Individual 1: 0.9 + 1.0 * 1  = 1.9
        // → Individual 1 should win
        let mut selector_high = FitnessConstraintsPenaltySurvival::new(1.0);
        let survived_high = selector_high.operate(pop, 1, &mut rng);
        assert_eq!(survived_high.fitness, array![0.9]);
    }
}
