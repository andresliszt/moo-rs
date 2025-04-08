use crate::genetic::IndividualGenes;
use crate::operators::{CrossoverOperator, GeneticOperator};
use crate::random::RandomGenerator;

#[derive(Clone, Debug)]
/// Uniform binary crossover operator for genetic algorithms.
pub struct UniformBinaryCrossover;

impl UniformBinaryCrossover {
    pub fn new() -> Self {
        Self {}
    }
}

impl GeneticOperator for UniformBinaryCrossover {
    fn name(&self) -> String {
        "UniformBinaryCrossover".to_string()
    }
}

impl CrossoverOperator for UniformBinaryCrossover {
    fn crossover(
        &self,
        parent_a: &IndividualGenes,
        parent_b: &IndividualGenes,
        rng: &mut dyn RandomGenerator,
    ) -> (IndividualGenes, IndividualGenes) {
        assert_eq!(
            parent_a.len(),
            parent_b.len(),
            "Parents must have the same number of genes"
        );

        let num_genes = parent_a.len();
        let mut offspring_a = IndividualGenes::zeros(num_genes);
        let mut offspring_b = IndividualGenes::zeros(num_genes);

        for i in 0..num_genes {
            if rng.gen_proability() < 0.5 {
                // Swap genes
                offspring_a[i] = parent_b[i];
                offspring_b[i] = parent_a[i];
            } else {
                // Keep genes
                offspring_a[i] = parent_a[i];
                offspring_b[i] = parent_b[i];
            }
        }

        (offspring_a, offspring_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{RandomGenerator, TestDummyRng};
    use ndarray::{Array1, array};
    use rand::RngCore;

    /// A controlled fake random generator that returns predetermined values for `gen_proability()`.
    struct ControlledFakeProbabilityGenerator {
        /// A list of predetermined f64 responses.
        responses: Vec<f64>,
        /// Current index into the responses vector.
        index: usize,
        /// Dummy RNG used to satisfy the `rng()` method.
        dummy: TestDummyRng,
    }

    impl ControlledFakeProbabilityGenerator {
        fn new(responses: Vec<f64>) -> Self {
            Self {
                responses,
                index: 0,
                dummy: TestDummyRng,
            }
        }
    }

    impl RandomGenerator for ControlledFakeProbabilityGenerator {
        fn rng(&mut self) -> &mut dyn RngCore {
            &mut self.dummy
        }
        // Override the default `gen_proability()` to return controlled values.
        fn gen_proability(&mut self) -> f64 {
            let value = self.responses[self.index];
            self.index += 1;
            value
        }
    }

    #[test]
    fn test_uniform_binary_crossover_controlled() {
        // Define two parent individuals.
        let parent_a: Array1<f64> = array![0.0, 1.0, 1.0, 0.0, 1.0];
        let parent_b: Array1<f64> = array![1.0, 0.0, 0.0, 1.0, 0.0];

        let crossover_operator = UniformBinaryCrossover::new();
        assert_eq!(crossover_operator.name(), "UniformBinaryCrossover");

        // Create a controlled fake RNG with predetermined probability values.
        // For each gene index, the decision is made by comparing the generated probability with 0.5:
        // Index 0: 0.6 (>= 0.5 → no swap)
        // Index 1: 0.7 (>= 0.5 → no swap)
        // Index 2: 0.8 (>= 0.5 → no swap)
        // Index 3: 0.3 (< 0.5 → swap)
        // Index 4: 0.4 (< 0.5 → swap)
        let responses = vec![0.6, 0.7, 0.8, 0.3, 0.4];
        let mut fake_rng = ControlledFakeProbabilityGenerator::new(responses);

        // Perform the uniform binary crossover.
        let (offspring_a, offspring_b) =
            crossover_operator.crossover(&parent_a, &parent_b, &mut fake_rng);

        // Expected offspring:
        // For index 0: no swap → offspring_a[0] = parent_a[0] = 0.0, offspring_b[0] = parent_b[0] = 1.0.
        // For index 1: no swap → offspring_a[1] = parent_a[1] = 1.0, offspring_b[1] = parent_b[1] = 0.0.
        // For index 2: no swap → offspring_a[2] = parent_a[2] = 1.0, offspring_b[2] = parent_b[2] = 0.0.
        // For index 3: swap    → offspring_a[3] = parent_b[3] = 1.0, offspring_b[3] = parent_a[3] = 0.0.
        // For index 4: swap    → offspring_a[4] = parent_b[4] = 0.0, offspring_b[4] = parent_a[4] = 1.0.
        let expected_offspring_a: Array1<f64> = array![0.0, 1.0, 1.0, 1.0, 0.0];
        let expected_offspring_b: Array1<f64> = array![1.0, 0.0, 0.0, 0.0, 1.0];

        assert_eq!(
            offspring_a, expected_offspring_a,
            "Offspring A is not as expected"
        );
        assert_eq!(
            offspring_b, expected_offspring_b,
            "Offspring B is not as expected"
        );
    }
}
