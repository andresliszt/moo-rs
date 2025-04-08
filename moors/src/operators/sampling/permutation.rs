use std::fmt::Debug;

use ndarray::Array1;

use crate::{
    genetic::IndividualGenes,
    operators::{GeneticOperator, SamplingOperator},
    random::RandomGenerator,
};

#[derive(Clone, Debug)]
/// Sampling operator for permutation-based variables.
pub struct PermutationSampling;

impl PermutationSampling {
    pub fn new() -> Self {
        Self
    }
}

impl GeneticOperator for PermutationSampling {
    fn name(&self) -> String {
        "PermutationSampling".to_string()
    }
}

impl SamplingOperator for PermutationSampling {
    /// Generates a single individual of length `n_vars` where the genes
    /// are a shuffled permutation of the integers [0, 1, 2, ..., n_vars - 1].
    fn sample_individual(&self, n_vars: usize, rng: &mut dyn RandomGenerator) -> IndividualGenes {
        // 1) Create a vector of indices [0, 1, 2, ..., n_vars - 1]
        let mut indices: Vec<f64> = (0..n_vars).map(|i| i as f64).collect();

        // 2) Shuffle the indices in-place using the `SliceRandom` trait
        rng.shuffle_vec(&mut indices);

        Array1::from_vec(indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{RandomGenerator, TestDummyRng};
    use rand::RngCore;

    /// A fake RandomGenerator for testing. It contains a TestDummyRng to provide
    /// the required RngCore behavior.
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
        fn rng(&mut self) -> &mut dyn RngCore {
            // Return a mutable reference to the internal dummy RNG.
            &mut self.dummy
        }

        fn shuffle_vec(&mut self, vector: &mut Vec<f64>) {
            vector.reverse();
        }
    }

    #[test]
    fn test_permutation_sampling_controlled() {
        // Create the sampling operator.
        let sampler = PermutationSampling;
        assert_eq!(sampler.name(), "PermutationSampling");
        // Use our fake RNG.
        let mut rng = FakeRandomGenerator::new();

        let population_size = 5;
        let n_vars = 4; // For example, 4 variables

        // Generate the population. It is assumed that `operate` (defined via
        // the SamplingOperator trait) generates population_size individuals.
        let population = sampler.operate(population_size, n_vars, &mut rng);

        // Check the population shape.
        assert_eq!(population.nrows(), population_size);
        assert_eq!(population.ncols(), n_vars);

        let expected = vec![3.0, 2.0, 1.0, 0.0];

        // Verify that each individual's genes match the expected permutation.
        for row in population.outer_iter() {
            let perm: Vec<f64> = row.to_vec();
            assert_eq!(
                perm, expected,
                "The permutation did not match the expected value."
            );
        }
    }
}
