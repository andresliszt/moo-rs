use crate::{
    genetic::IndividualGenesMut,
    operators::{GeneticOperator, MutationOperator},
    random::RandomGenerator,
};

#[derive(Clone, Debug)]
/// Mutation operator that swaps two genes in a permutation-based individual.
pub struct SwapMutation;

impl SwapMutation {
    pub fn new() -> Self {
        Self
    }
}

impl GeneticOperator for SwapMutation {
    fn name(&self) -> String {
        "SwapMutation".to_string()
    }
}

/// In a typical permutation-based setup, each row is an array of distinct values.
/// The "swap" mutation picks two indices at random and swaps them.
impl MutationOperator for SwapMutation {
    fn mutate<'a>(&self, mut individual: IndividualGenesMut<'a>, rng: &mut dyn RandomGenerator) {
        let length = individual.len();
        // If there is at most one element, there's nothing to swap.
        if length > 1 {
            // Pick two distinct indices.
            let idx1 = rng.gen_range_usize(0, length);
            let mut idx2 = rng.gen_range_usize(0, length);
            while idx2 == idx1 {
                idx2 = rng.gen_range_usize(0, length);
            }

            // Swap the elements in place.
            individual.swap(idx1, idx2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genetic::PopulationGenes;
    use crate::random::{RandomGenerator, TestDummyRng};
    use ndarray::array;
    use rand::RngCore;

    /// A controlled fake RandomGenerator that returns predetermined values for `gen_range_usize`.
    struct ControlledFakeRandomGenerator {
        responses: Vec<usize>,
        index: usize,
        dummy: TestDummyRng,
    }

    impl ControlledFakeRandomGenerator {
        /// Create a new controlled fake with a vector of responses.
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
            // Return the next predetermined response.
            let resp = self.responses[self.index];
            self.index += 1;
            resp
        }
    }

    #[test]
    fn test_swap_mutation_controlled() {
        // Create an individual: [0, 1, 2, 3, 4]
        let mut original: PopulationGenes = array![[0.0, 1.0, 2.0, 3.0, 4.0]];
        let original_row = original.row(0).to_owned();

        // Create a SwapMutation operator.
        let mutation_operator = SwapMutation::new();
        assert_eq!(mutation_operator.name(), "SwapMutation");

        // Create a controlled fake RNG that returns 1 on the first call and 3 on the second call.
        // This means idx1 will be 1 and idx2 will be 3.
        let mut rng = ControlledFakeRandomGenerator::new(vec![1, 3]);

        // Mutate the individual.
        mutation_operator.operate(&mut original, 1.0, &mut rng);

        // Since there is only one row, get that individual.
        let mutated_individual = original.row(0).to_owned();

        // Determine which positions differ between the original and mutated individual.
        let mut diff_positions = Vec::new();
        for i in 0..original_row.len() {
            if original_row[i] != mutated_individual[i] {
                diff_positions.push(i);
            }
        }

        // Exactly 2 positions should have changed after one swap.
        assert_eq!(
            diff_positions.len(),
            2,
            "Expected exactly 2 positions to change after swap"
        );

        // Verify that the elements at those two positions have been swapped.
        let i = diff_positions[0];
        let j = diff_positions[1];
        assert_eq!(
            original_row[i], mutated_individual[j],
            "Element at position {} should have been swapped with element at position {}",
            i, j
        );
        assert_eq!(
            original_row[j], mutated_individual[i],
            "Element at position {} should have been swapped with element at position {}",
            j, i
        );
    }
}
