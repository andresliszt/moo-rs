use ndarray::{Array1, Array2};

use crate::random::RandomGenerator;

mod permutation;
mod random;

pub use permutation::PermutationSampling;
pub use random::{PerGeneSampling, RandomSamplingBinary, RandomSamplingFloat, RandomSamplingInt};

pub trait SamplingOperator {
    /// Samples a single individual.
    fn sample_individual(&self, num_vars: usize, rng: &mut impl RandomGenerator) -> Array1<f64>;

    /// Samples a population of individuals.
    fn operate(
        &self,
        population_size: usize,
        num_vars: usize,
        rng: &mut impl RandomGenerator,
    ) -> Array2<f64> {
        let mut population = Vec::with_capacity(population_size);

        // Sample individuals and collect them
        for _ in 0..population_size {
            let individual = self.sample_individual(num_vars, rng);
            population.push(individual);
        }

        // Determine the number of genes per individual
        let num_genes = population[0].len();

        // Flatten the population into a single vector
        let flat_population: Vec<f64> = population
            .into_iter()
            .flat_map(|individual| individual.into_iter())
            .collect();

        // Create the shape: (number of individuals, number of genes)
        let shape = (population_size, num_genes);

        // Use from_shape_vec to create PopulationGenes
        Array2::<f64>::from_shape_vec(shape, flat_population)
            .expect("Failed to create PopulationGenes from vector")
    }
}
