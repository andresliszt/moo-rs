use ndarray::{Array1, Axis, concatenate, s};

use crate::genetic::IndividualGenes;
use crate::operators::{CrossoverOperator, GeneticOperator};
use crate::random::RandomGenerator;

#[derive(Clone, Debug)]
/// Single-point crossover operator for binary-encoded individuals.
pub struct SinglePointBinaryCrossover;

impl SinglePointBinaryCrossover {
    pub fn new() -> Self {
        Self {}
    }
}

impl GeneticOperator for SinglePointBinaryCrossover {
    fn name(&self) -> String {
        "SinglePointBinaryCrossover".to_string()
    }
}

impl CrossoverOperator for SinglePointBinaryCrossover {
    fn crossover(
        &self,
        parent_a: &IndividualGenes,
        parent_b: &IndividualGenes,
        rng: &mut dyn RandomGenerator,
    ) -> (IndividualGenes, IndividualGenes) {
        let num_genes = parent_a.len();
        assert_eq!(
            num_genes,
            parent_b.len(),
            "Parents must have the same number of genes"
        );

        if num_genes == 0 {
            return (Array1::default(0), Array1::default(0));
        }

        // Choose a crossover point between 1 and num_genes - 1
        let crossover_point = rng.gen_range_usize(1, num_genes);

        // Split parents at the crossover point and create offspring
        let offspring_a = concatenate![
            Axis(0),
            parent_a.slice(s![..crossover_point]),
            parent_b.slice(s![crossover_point..])
        ];

        let offspring_b = concatenate![
            Axis(0),
            parent_b.slice(s![..crossover_point]),
            parent_a.slice(s![crossover_point..])
        ];

        (offspring_a, offspring_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use ndarray::array;
    use rand::RngCore;

    use crate::random::{RandomGenerator, TestDummyRng};

    /// A controlled fake RandomGenerator that returns predetermined values for `gen_range_usize`.
    struct ControlledFakeRandomGenerator {
        responses: Vec<usize>,
        index: usize,
        dummy: TestDummyRng,
    }

    impl ControlledFakeRandomGenerator {
        fn new(responses: Vec<usize>) -> Self {
            Self {
                responses,
                index: 0,
                dummy: TestDummyRng,
            }
        }
    }

    impl RandomGenerator for ControlledFakeRandomGenerator {
        fn rng(&mut self) -> &mut dyn RngCore {
            &mut self.dummy
        }
        fn gen_range_usize(&mut self, _min: usize, _max: usize) -> usize {
            let resp = self.responses[self.index];
            self.index += 1;
            resp
        }
    }

    #[test]
    fn test_single_point_binary_crossover_controlled() {
        // Define two binary-encoded parents.
        let parent_a: Array1<f64> = array![0.0, 1.0, 1.0, 0.0, 1.0];
        let parent_b: Array1<f64> = array![1.0, 0.0, 0.0, 1.0, 0.0];

        // Create the SinglePointBinaryCrossover operator.
        let crossover_operator = SinglePointBinaryCrossover::new();
        assert_eq!(crossover_operator.name(), "SinglePointBinaryCrossover");

        // Use a controlled fake RNG that always returns 3 as the crossover point.
        // The call to `gen_range_usize(1, num_genes)` will return 3.
        let mut fake_rng = ControlledFakeRandomGenerator::new(vec![3]);

        // Perform the crossover.
        let (offspring_a, offspring_b) =
            crossover_operator.crossover(&parent_a, &parent_b, &mut fake_rng);

        // With a crossover point of 3:
        // offspring_a = [parent_a[0..3], parent_b[3..]] = [0.0, 1.0, 1.0, 1.0, 0.0]
        // offspring_b = [parent_b[0..3], parent_a[3..]] = [1.0, 0.0, 0.0, 0.0, 1.0]
        let expected_offspring_a = array![0.0, 1.0, 1.0, 1.0, 0.0];
        let expected_offspring_b = array![1.0, 0.0, 0.0, 0.0, 1.0];

        // Assert that the resulting offspring match the expected outcomes.
        assert_eq!(
            offspring_a, expected_offspring_a,
            "Offspring A did not match the expected output"
        );
        assert_eq!(
            offspring_b, expected_offspring_b,
            "Offspring B did not match the expected output"
        );
    }
}
