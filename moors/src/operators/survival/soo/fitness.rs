use ndarray::Array1;

use crate::{
    algorithms::AlgorithmContext,
    genetic::{D12, PopulationSOO},
    operators::survival::SurvivalOperator,
    random::RandomGenerator,
};
/// `FitnessSurvival` selects the individuals with the lowest fitness values
/// (minimization) to survive into the next generation.
pub struct FitnessSurvival {}

impl FitnessSurvival {
    pub fn new() -> Self {
        Self {}
    }
}

impl SurvivalOperator for FitnessSurvival {
    type FDim = ndarray::Ix1;

    fn operate<ConstrDim>(
        &mut self,
        population: PopulationSOO<ConstrDim>,
        num_survive: usize,
        _rng: &mut impl RandomGenerator,
        _algorithm_context: &AlgorithmContext,
    ) -> PopulationSOO<ConstrDim>
    where
        ConstrDim: D12,
    {
        let pop_size = population.fitness.len();
        let mut indices: Vec<usize> = (0..pop_size).collect();

        indices.sort_by(|&i, &j| {
            population.fitness[i]
                .partial_cmp(&population.fitness[j])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let survive_count = num_survive.min(pop_size);
        let selected_indices = &indices[..survive_count];
        let mut selected_population = population.selected(selected_indices);
        // Population is already sorted by fitness
        selected_population.set_rank(Array1::from_iter(0..survive_count));
        selected_population
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::random::TestDummyRng;
    use ndarray::{Array1, Array2, array};

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

    /// Helper to create a PopulationSOO from a slice of fitness values.
    /// Genes are dummy zeros (shape (pop_size, 1)), since FitnessSurvival only reads `.fitness`.
    fn make_population(fitness_values: &[f64]) -> PopulationSOO<ndarray::Ix2> {
        let pop_size = fitness_values.len();
        let genes = Array2::zeros((pop_size, 1));
        let fitness = Array1::from_iter(fitness_values.iter().cloned());
        PopulationSOO::new_unconstrained(genes, fitness)
    }

    #[test]
    fn selects_lowest_two_fitness() {
        // Four individuals with fitness [0.8, 0.2, 0.5, 0.1].
        // Selecting 2 should keep indices 3 (0.1) and 1 (0.2).
        let pop = make_population(&[0.8, 0.2, 0.5, 0.1]);
        let mut rng = FakeRandomGenerator::new();
        // create context (not used in the algorithm)
        let _context = AlgorithmContext::new(1, 4, 4, 1, 1, 0, None, None);

        let mut selector = FitnessSurvival::new();
        let survived = selector.operate(pop, 2, &mut rng, &_context);

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
    fn selects_all_when_num_survive_exceeds_population() {
        // Three individuals [0.3, 0.1, 0.2], requesting 5 survivors
        // should return all 3 sorted by fitness: [0.1, 0.2, 0.3]
        let pop = make_population(&[0.3, 0.1, 0.2]);
        let mut rng = FakeRandomGenerator::new();
        // create context (not used in the algorithm)
        let _context = AlgorithmContext::new(1, 3, 4, 1, 1, 0, None, None);

        let mut selector = FitnessSurvival::new();
        let survived = selector.operate(pop, 5, &mut rng, &_context);

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
        let pop = make_population(&[0.42]);
        let mut rng = FakeRandomGenerator::new();
        // create context (not used in the algorithm)
        let _context = AlgorithmContext::new(1, 1, 1, 1, 1, 0, None, None);

        let mut selector = FitnessSurvival::new();
        let survived = selector.operate(pop, 1, &mut rng, &_context);

        assert_eq!(survived.fitness.len(), 1);
        assert_eq!(survived.fitness[0], 0.42);

        // Only one rank: [0]
        let expected_ranks = array![0];
        assert_eq!(survived.rank.unwrap(), expected_ranks);
    }
}
