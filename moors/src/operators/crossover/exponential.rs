use crate::genetic::IndividualGenes;
use crate::operators::{CrossoverOperator, GeneticOperator};
use crate::random::RandomGenerator;

#[derive(Clone, Debug)]
/// Crossover operator that combines parent genes based on an exponential distribution.
pub struct ExponentialCrossover {
    pub exponential_crossover_rate: f64,
}

impl ExponentialCrossover {
    pub fn new(exponential_crossover_rate: f64) -> Self {
        Self {
            exponential_crossover_rate,
        }
    }
}

impl GeneticOperator for ExponentialCrossover {
    fn name(&self) -> String {
        format!(
            "ExponentialCrossover(exponential_crossover_rate={})",
            self.exponential_crossover_rate
        )
    }
}

impl CrossoverOperator for ExponentialCrossover {
    fn crossover(
        &self,
        parent_a: &IndividualGenes,
        parent_b: &IndividualGenes,
        rng: &mut dyn RandomGenerator,
    ) -> (IndividualGenes, IndividualGenes) {
        let len = parent_a.len();
        assert_eq!(len, parent_b.len());

        // We'll do "child_a" from (A,B) and "child_b" from (B,A).
        let mut child_a = parent_a.clone();
        let mut child_b = parent_b.clone();

        // random start [0..len)
        let start = rng.gen_range_usize(0, len);

        // We'll copy from parent_b into child_a as long as rng < CR
        // in a circular manner
        let mut i = start;
        loop {
            child_a[i] = parent_b[i];
            // Next index (wrap around)
            i = (i + 1) % len;
            if i == start {
                break; // completed a full circle, stop
            }
            let r: f64 = rng.gen_proability();
            if r >= self.exponential_crossover_rate {
                break;
            }
        }

        // Similarly for child_b, but swap roles of A/B
        let start_b = rng.gen_range_usize(0, len);
        let mut j = start_b;
        loop {
            child_b[j] = parent_a[j];
            j = (j + 1) % len;
            if j == start_b {
                break;
            }
            let r: f64 = rng.gen_proability();
            if r >= self.exponential_crossover_rate {
                break;
            }
        }

        (child_a, child_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{RandomGenerator, TestDummyRng};
    use ndarray::array;

    /// A simple fake random generator for controlled testing of ExponentialCrossover.
    /// It returns predetermined values for `gen_range_usize` and `gen_proability`.
    struct FakeRandom {
        /// Predefined responses for calls to `gen_range_usize`.
        range_values: Vec<usize>,
        /// Predefined responses for calls to `gen_proability`.
        probability_values: Vec<f64>,
        /// Dummy RNG to satisfy the trait requirement.
        dummy: TestDummyRng,
    }

    impl FakeRandom {
        /// Creates a new instance with the given responses.
        fn new(range_values: Vec<usize>, probability_values: Vec<f64>) -> Self {
            Self {
                range_values,
                probability_values,
                dummy: TestDummyRng,
            }
        }
    }

    impl RandomGenerator for FakeRandom {
        /// Returns a mutable reference to the underlying dummy RNG.
        fn rng(&mut self) -> &mut dyn rand::RngCore {
            &mut self.dummy
        }

        /// Returns the next predetermined value for a range selection.
        fn gen_range_usize(&mut self, _min: usize, _max: usize) -> usize {
            self.range_values.remove(0)
        }

        /// Returns the next predetermined probability value.
        fn gen_proability(&mut self) -> f64 {
            self.probability_values.remove(0)
        }
    }

    #[test]
    fn test_exponential_crossover() {
        // Define two parent IndividualGenes as small Array1<f64> vectors.
        // For simplicity, we use arrays of length 3.
        let parent_a: IndividualGenes = array![1.0, 2.0, 3.0];
        let parent_b: IndividualGenes = array![4.0, 5.0, 6.0];

        // Create the ExponentialCrossover operator with a crossover rate of 0.5.
        let operator = ExponentialCrossover::new(0.5);
        assert_eq!(
            operator.name(),
            "ExponentialCrossover(exponential_crossover_rate=0.5)"
        );

        // Set up the fake random generator:
        // - For child_a, gen_range_usize returns 1 (start index = 1) and then
        //   gen_proability returns 0.7 (>= 0.5) so the loop stops after one replacement.
        // - For child_b, gen_range_usize returns 2 (start index = 2) and then
        //   gen_proability returns 0.8 (>= 0.5) so the loop stops after one replacement.
        let mut fake_rng = FakeRandom::new(vec![1, 2], vec![0.7, 0.8]);

        // Perform the exponential crossover.
        let (child_a, child_b) = operator.crossover(&parent_a, &parent_b, &mut fake_rng);

        // Expected outcome:
        // For child_a: start index 1 -> replace gene at index 1 with parent_b[1] (5.0),
        // so child_a becomes [1.0, 5.0, 3.0].
        let expected_child_a: IndividualGenes = array![1.0, 5.0, 3.0];

        // For child_b: start index 2 -> replace gene at index 2 with parent_a[2] (3.0),
        // so child_b becomes [4.0, 5.0, 3.0].
        let expected_child_b: IndividualGenes = array![4.0, 5.0, 3.0];

        // Check that the results match the expectations.
        assert_eq!(
            child_a, expected_child_a,
            "child_a does not match expected output"
        );
        assert_eq!(
            child_b, expected_child_b,
            "child_b does not match expected output"
        );
    }
}
