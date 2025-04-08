use ndarray::Axis;

use crate::{
    genetic::{IndividualGenesMut, PopulationGenes},
    operators::GeneticOperator,
    random::RandomGenerator,
};

pub mod bitflip;
pub mod displacement;
pub mod gaussian;
pub mod scramble;
pub mod swap;

/// MutationOperator defines an in-place mutation where the individual is modified directly.
pub trait MutationOperator: GeneticOperator {
    /// Mutates a single individual in place.
    ///
    /// # Arguments
    ///
    /// * `individual` - The individual to mutate, provided as a mutable view.
    /// * `rng` - A random number generator.
    fn mutate<'a>(&self, individual: IndividualGenesMut<'a>, rng: &mut dyn RandomGenerator);

    /// Selects individuals for mutation based on the mutation rate.
    fn select_individuals_for_mutation(
        &self,
        population_size: usize,
        mutation_rate: f64,
        rng: &mut dyn RandomGenerator,
    ) -> Vec<bool> {
        (0..population_size)
            .map(|_| rng.gen_bool(mutation_rate))
            .collect()
    }

    /// Applies the mutation operator to the entire population in place.
    ///
    /// # Arguments
    ///
    /// * `population` - The population as a mutable 2D array (each row represents an individual).
    /// * `mutation_rate` - The probability that an individual is mutated.
    /// * `rng` - A random number generator.
    fn operate(
        &self,
        population: &mut PopulationGenes,
        mutation_rate: f64,
        rng: &mut dyn RandomGenerator,
    ) {
        // Get the number of individuals (i.e. the number of rows).
        let population_size = population.len_of(Axis(0));
        // Generate a boolean mask for which individuals will be mutated.
        let mask: Vec<bool> =
            self.select_individuals_for_mutation(population_size, mutation_rate, rng);

        // Iterate over the population using outer_iter_mut to get a mutable view for each row.
        for (i, mut individual) in population.outer_iter_mut().enumerate() {
            if mask[i] {
                // Pass a mutable view of the individual to the mutate method.
                self.mutate(individual.view_mut(), rng);
            }
        }
    }
}
