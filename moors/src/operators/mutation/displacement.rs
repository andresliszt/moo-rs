use ndarray::{Array1, Axis, concatenate, s};

use crate::{
    genetic::IndividualGenesMut,
    operators::{GeneticOperator, MutationOperator},
    random::RandomGenerator,
};

#[derive(Clone, Debug)]
/// Displacement Mutation operator that extracts a segment from the chromosome
/// and reinserts it at a different random position using ndarray operations.
pub struct DisplacementMutation {}

impl DisplacementMutation {
    pub fn new() -> Self {
        Self {}
    }
}

impl GeneticOperator for DisplacementMutation {
    fn name(&self) -> String {
        "DisplacementMutation".to_string()
    }
}

impl MutationOperator for DisplacementMutation {
    fn mutate<'a>(&self, mut individual: IndividualGenesMut<'a>, rng: &mut dyn RandomGenerator) {
        let n = individual.len();

        // Select two random indices to define the segment boundaries.
        let idx1 = rng.gen_range_usize(0, n);
        let idx2 = rng.gen_range_usize(0, n);
        let (start, end) = if idx1 <= idx2 {
            (idx1, idx2)
        } else {
            (idx2, idx1)
        };

        // If the indices are equal, there is no segment to displace.
        if start == end {
            return;
        }

        // Split the individual into three parts: left, segment, and right.
        let left: Array1<f64> = individual.slice(s![0..start]).to_owned();
        let segment: Array1<f64> = individual.slice(s![start..end]).to_owned();
        let right: Array1<f64> = individual.slice(s![end..]).to_owned();

        // Combine left and right to form the remainder.
        let remainder = concatenate![Axis(0), left, right];
        let remainder_len = remainder.len();

        // Choose a new insertion index in the remainder.
        let new_index = rng.gen_range_usize(0, remainder_len + 1);

        // Split the remainder into two parts.
        let remainder_left = remainder.slice(s![0..new_index]).to_owned();
        let remainder_right = remainder.slice(s![new_index..]).to_owned();

        // Reconstruct the mutated individual by concatenating remainder_left, segment, and remainder_right.
        let new_individual = concatenate![Axis(0), remainder_left, segment, remainder_right];

        // Copy the result back into the original individual view.
        individual.assign(&new_individual);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{RandomGenerator, TestDummyRng};
    use ndarray::{Array1, array};
    use rstest::rstest;

    /// A fake RandomGenerator for testing that returns predefined usize values.
    struct FakeRandomGeneratorDisplacement {
        usize_values: Vec<usize>,
        dummy: TestDummyRng,
    }

    impl FakeRandomGeneratorDisplacement {
        fn new(usize_values: Vec<usize>) -> Self {
            Self {
                usize_values,
                dummy: TestDummyRng,
            }
        }
    }

    impl RandomGenerator for FakeRandomGeneratorDisplacement {
        fn rng(&mut self) -> &mut dyn rand::RngCore {
            &mut self.dummy
        }
        fn gen_range_usize(&mut self, _min: usize, _max: usize) -> usize {
            self.usize_values.remove(0)
        }
    }

    #[rstest(rng_values,
        case(vec![2, 5, 1]),
        case(vec![5, 2, 1])
    )]
    fn test_displacement_mutation(rng_values: Vec<usize>) {
        // Test individual: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        let mut individual: Array1<f64> = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

        {
            // Create a mutable view of the individual.
            let view = individual.view_mut();

            // Create a FakeRandomGenerator with the parameterized values.
            let mut rng = FakeRandomGeneratorDisplacement::new(rng_values);

            let mutation_operator = DisplacementMutation::new();
            assert_eq!(mutation_operator.name(), "DisplacementMutation");
            mutation_operator.mutate(view, &mut rng);
        }

        // Expected result after mutation: [0.0, 2.0, 3.0, 4.0, 1.0, 5.0]
        let expected: Array1<f64> = array![0.0, 2.0, 3.0, 4.0, 1.0, 5.0];
        assert_eq!(individual, expected);
    }

    #[test]
    fn test_displacement_mutation_same_idx() {
        // in this test gen range return 0 twice, so no segment...
        let mut individual: Array1<f64> = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

        {
            // Create a mutable view of the individual.
            let view = individual.view_mut();

            // Create a FakeRandomGenerator with the parameterized values.
            let mut rng = FakeRandomGeneratorDisplacement::new(vec![0, 0]);

            let mutation_operator = DisplacementMutation::new();
            mutation_operator.mutate(view, &mut rng);
        }

        let expected: Array1<f64> = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(individual, expected);
    }
}
