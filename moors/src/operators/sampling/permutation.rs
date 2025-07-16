use ndarray::Array1;

use crate::{operators::SamplingOperator, random::RandomGenerator};

#[derive(Debug, Clone)]
/// Sampling operator for permutation-based variables.
pub struct PermutationSampling;

impl PermutationSampling {
    pub fn new() -> Self {
        Self
    }
}

impl Default for PermutationSampling {
    fn default() -> Self {
        Self::new()
    }
}

impl SamplingOperator for PermutationSampling {
    /// Generates a single individual of length `num_vars` where the genes
    /// are a shuffled permutation of the integers [0, 1, 2, ..., num_vars - 1].
    fn sample_individual(&self, num_vars: usize, rng: &mut impl RandomGenerator) -> Array1<f64> {
        // 1) Create a vector of indices [0, 1, 2, ..., num_vars - 1]
        let mut indices: Vec<f64> = (0..num_vars).map(|i| i as f64).collect();

        // 2) Shuffle the indices in-place using the `SliceRandom` trait
        rng.shuffle_vec(&mut indices);

        Array1::from_vec(indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{RandomGenerator, TestDummyRng};

    /// A fake RandomGenerator for testing. It contains a TestDummyRng to provide
    /// the required RandomGenerator behavior.
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
        // Use our fake RNG.
        let mut rng = FakeRandomGenerator::new();

        let population_size = 5;
        let num_vars = 4; // For example, 4 variables

        // Generate the population. It is assumed that `operate` (defined via
        // the SamplingOperator trait) generates population_size individuals.
        let population = sampler.operate(population_size, num_vars, &mut rng);

        // Check the population shape.
        assert_eq!(population.nrows(), population_size);
        assert_eq!(population.ncols(), num_vars);

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
