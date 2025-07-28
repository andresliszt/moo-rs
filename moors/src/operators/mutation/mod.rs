use ndarray::{Array2, ArrayViewMut1, Axis};

use crate::random::RandomGenerator;

mod bitflip;
mod displacement;
mod gaussian;
mod inversion;
mod polynomial;
mod scramble;
mod swap;
mod uniform;

pub use bitflip::BitFlipMutation;
pub use displacement::DisplacementMutation;
pub use gaussian::GaussianMutation;
pub use inversion::InversionMutation;
pub use polynomial::PolynomialMutation;
pub use scramble::ScrambleMutation;
pub use swap::SwapMutation;
pub use uniform::{UniformBinaryMutation, UniformRealMutation};

/// MutationOperator defines an in-place mutation where the individual is modified directly.
pub trait MutationOperator {
    /// Mutates a single individual in place.
    ///
    /// # Arguments
    ///
    /// * `individual` - The individual to mutate, provided as a mutable view.
    /// * `rng` - A random number generator.
    fn mutate<'a>(&self, individual: ArrayViewMut1<'a, f64>, rng: &mut impl RandomGenerator);

    /// Selects individuals for mutation based on the mutation rate.
    fn select_individuals_for_mutation(
        &self,
        population_size: usize,
        mutation_rate: f64,
        rng: &mut impl RandomGenerator,
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
        population: &mut Array2<f64>,
        mutation_rate: f64,
        rng: &mut impl RandomGenerator,
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
