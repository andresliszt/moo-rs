use ndarray::Array1;

use crate::genetic::IndividualGenes;
use crate::operators::{CrossoverOperator, GeneticOperator};
use crate::random::RandomGenerator;

#[derive(Clone, Debug)]
/// Crossover operator for permutation-based individuals using Order Crossover (OX).
pub struct OrderCrossover;

impl OrderCrossover {
    pub fn new() -> Self {
        Self {}
    }
}

impl GeneticOperator for OrderCrossover {
    fn name(&self) -> String {
        "OrderCrossover".to_string()
    }
}

impl CrossoverOperator for OrderCrossover {
    fn crossover(
        &self,
        parent_a: &IndividualGenes,
        parent_b: &IndividualGenes,
        rng: &mut dyn RandomGenerator,
    ) -> (IndividualGenes, IndividualGenes) {
        let len = parent_a.len();
        assert_eq!(len, parent_b.len());

        // Choose 2 cut points
        let mut p1 = rng.gen_range_usize(0, len);
        let mut p2 = rng.gen_range_usize(0, len);
        if p1 > p2 {
            std::mem::swap(&mut p1, &mut p2);
        }

        // Initialize children with some default
        let mut child_a = Array1::from_elem(len, f64::NAN);
        let mut child_b = Array1::from_elem(len, f64::NAN);

        // copy [p1..p2] from A-> childA, B-> childB
        for i in p1..p2 {
            child_a[i] = parent_a[i];
            child_b[i] = parent_b[i];
        }

        // Fill remainder of child_a from parent_b in order
        {
            let mut fill_index = p2 % len; // start filling after p2
            for i in 0..len {
                // index in parent_b
                let idx_b = (p2 + i) % len;
                let val_b = parent_b[idx_b];
                // skip if already in child_a
                if !child_a.iter().any(|&x| x == val_b) {
                    child_a[fill_index] = val_b;
                    fill_index = (fill_index + 1) % len;
                }
            }
        }

        // Fill remainder of child_b from parent_a in order
        {
            let mut fill_index = p2 % len;
            for i in 0..len {
                let idx_a = (p2 + i) % len;
                let val_a = parent_a[idx_a];
                if !child_b.iter().any(|&x| x == val_a) {
                    child_b[fill_index] = val_a;
                    fill_index = (fill_index + 1) % len;
                }
            }
        }

        (child_a, child_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genetic::IndividualGenes;
    use crate::random::{RandomGenerator, TestDummyRng};
    use ndarray::Array1;
    use rand::RngCore;

    /// A controlled fake RandomGenerator that returns predetermined values for `gen_range_usize`.
    /// For this test, it will return 2 on the first call and 5 on the second call.
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
    fn test_order_crossover_controlled() {
        let len = 8;
        // Define parent_a as [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        let parent_a: IndividualGenes = Array1::from_vec((0..len).map(|x| x as f64).collect());
        // Define parent_b as [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        let parent_b: IndividualGenes =
            Array1::from_vec((0..len).map(|x| (len - 1 - x) as f64).collect());

        // Create the OrderCrossover operator.
        let crossover_operator = OrderCrossover::new();
        assert_eq!(crossover_operator.name(), "OrderCrossover");

        // Create a controlled fake RNG with predetermined responses [2, 5]
        // which means p1 = 2 and p2 = 5.
        let mut fake_rng = ControlledFakeRandomGenerator::new(vec![2, 5]);

        // Perform the crossover.
        let (child_a, child_b) = crossover_operator.crossover(&parent_a, &parent_b, &mut fake_rng);

        // Expected offspring:
        //
        // With p1 = 2 and p2 = 5:
        // - For child_a:
        //   - Copy indices 2..5 from parent_a: [2.0, 3.0, 4.0]
        //   - Fill the remainder from parent_b (starting from index 5) in order:
        //     Parent_b = [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        //     Skipping already present values yields:
        //     [6.0, 5.0, 2.0, 3.0, 4.0, 1.0, 0.0, 7.0]
        //
        // - For child_b:
        //   - Copy indices 2..5 from parent_b: [5.0, 4.0, 3.0]
        //   - Fill the remainder from parent_a (starting from index 5) in order:
        //     Parent_a = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        //     Skipping already present values yields:
        //     [1.0, 2.0, 5.0, 4.0, 3.0, 6.0, 7.0, 0.0]
        let expected_child_a = Array1::from_vec(vec![6.0, 5.0, 2.0, 3.0, 4.0, 1.0, 0.0, 7.0]);
        let expected_child_b = Array1::from_vec(vec![1.0, 2.0, 5.0, 4.0, 3.0, 6.0, 7.0, 0.0]);

        // Assert that the produced children match the expected results.
        assert_eq!(
            child_a, expected_child_a,
            "Child A does not match the expected output"
        );
        assert_eq!(
            child_b, expected_child_b,
            "Child B does not match the expected output"
        );
    }
}
