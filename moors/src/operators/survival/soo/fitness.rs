use std::cmp::Ordering;

use ndarray::Array1;

use crate::{
    genetic::{D12, PopulationSOO},
    operators::survival::SurvivalOperator,
    random::RandomGenerator,
};

/// A survival operator that selects individuals with the lowest fitness values
/// (for minimization problems) to survive into the next generation.
///
/// If constraint violations are present in the population, selection is based on a
/// **lexicographic sort**: individuals are prioritized first by their **total constraint
/// violation**, and then — in case of ties — by their **fitness** values.
///
/// - Individuals with **lower constraint violations** are preferred.
/// - Among individuals with equal violation, those with **lower fitness** are selected.
///
/// If no constraint violations are provided, selection is based solely on fitness.
///
/// This operator is used in single-objective optimization scenarios.
pub struct FitnessSurvival;

impl SurvivalOperator for FitnessSurvival {
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
            // Lexicographic sort: primary by constraint violations, secondary by fitness
            indices.sort_by(|&i, &j| {
                let ord1 = violations[i]
                    .partial_cmp(&violations[j])
                    .unwrap_or(Ordering::Equal);
                if ord1 != Ordering::Equal {
                    ord1
                } else {
                    population.fitness[i]
                        .partial_cmp(&population.fitness[j])
                        .unwrap_or(Ordering::Equal)
                }
            });
        } else {
            // Sort only by fitness
            indices.sort_by(|&i, &j| {
                population.fitness[i]
                    .partial_cmp(&population.fitness[j])
                    .unwrap_or(Ordering::Equal)
            });
        }
        let survive_count = num_survive.min(pop_size);
        let selected_indices = &indices[..survive_count];
        let mut selected_population = population.selected(selected_indices);
        // Population is already lex-sorted by cv and fitness
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
        let mut selector = FitnessSurvival;
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
        let mut selector = FitnessSurvival;
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
    fn selects_all_when_num_survive_exceeds_population() {
        // Three individuals [0.3, 0.1, 0.2], requesting 5 survivors
        // should return all 3 sorted by fitness: [0.1, 0.2, 0.3]
        let genes = Array2::zeros((4, 1));
        let fitness = array![0.3, 0.1, 0.2];
        let pop = PopulationSOO::new_unconstrained(genes, fitness);
        let mut rng = FakeRandomGenerator::new();
        let mut selector = FitnessSurvival;
        let survived = selector.operate(pop, 5, &mut rng);

        assert_eq!(survived.fitness.len(), 3);
        let expected_fitness = array![0.1, 0.2, 0.3];
        assert_eq!(survived.fitness, expected_fitness);
        // Expected ranks: [0, 1, 2]
        let expected_ranks = array![0, 1, 2];
        assert_eq!(survived.rank.unwrap(), expected_ranks);
    }

    #[test]
    fn single_individual_survives_with_rank_zero() {
        // One individual with fitness [0.42], selecting 1 → same individual
        let genes = Array2::zeros((1, 1));
        let fitness = array![0.42];
        let pop = PopulationSOO::new_unconstrained(genes, fitness);
        let mut rng = FakeRandomGenerator::new();
        let mut selector = FitnessSurvival;
        let survived = selector.operate(pop, 1, &mut rng);

        assert_eq!(survived.fitness.len(), 1);
        assert_eq!(survived.fitness[0], 0.42);

        // Only one rank: [0]
        let expected_ranks = array![0];
        assert_eq!(survived.rank.unwrap(), expected_ranks);
    }
}
