use crate::{
    genetic::{IndividualGenes, PopulationGenes},
    operators::GeneticOperator,
    random::RandomGenerator,
};

mod permutation;
mod random;

pub use permutation::PermutationSampling;
pub use random::{RandomSamplingBinary, RandomSamplingFloat, RandomSamplingInt};

pub trait SamplingOperator: GeneticOperator {
    /// Samples a single individual.
    fn sample_individual(&self, n_vars: usize, rng: &mut dyn RandomGenerator) -> IndividualGenes;

    /// Samples a population of individuals.
    fn operate(
        &self,
        population_size: usize,
        n_vars: usize,
        rng: &mut dyn RandomGenerator,
    ) -> PopulationGenes {
        let mut population = Vec::with_capacity(population_size);

        // Sample individuals and collect them
        for _ in 0..population_size {
            let individual = self.sample_individual(n_vars, rng);
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
        let population_genes = PopulationGenes::from_shape_vec(shape, flat_population)
            .expect("Failed to create PopulationGenes from vector");

        population_genes
    }
}
