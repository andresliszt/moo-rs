use ndarray::s;

use crate::{
    genetic::IndividualGenesMut,
    operators::{GeneticOperator, MutationOperator},
    random::RandomGenerator,
};

#[derive(Clone, Debug)]
/// Scramble Mutation operator that selects a subset of the chromosome and randomly reorders it.
pub struct ScrambleMutation {}

impl ScrambleMutation {
    pub fn new() -> Self {
        Self {}
    }
}

impl GeneticOperator for ScrambleMutation {
    fn name(&self) -> String {
        "ScrambleMutation".to_string()
    }
}

impl MutationOperator for ScrambleMutation {
    fn mutate<'a>(&self, mut individual: IndividualGenesMut<'a>, rng: &mut dyn RandomGenerator) {
        let n = individual.len();
        // Select two random indices to define the segment.
        let idx1 = rng.gen_range_usize(0, n);
        let idx2 = rng.gen_range_usize(0, n);
        let (start, end) = if idx1 <= idx2 {
            (idx1, idx2)
        } else {
            (idx2, idx1)
        };
        if start == end {
            return;
        }
        // Extract the segment.
        let mut segment = individual.slice(s![start..end]).to_vec();
        // Randomly reorder (scramble) the segment.
        rng.shuffle_vec(&mut segment);
        // Copy the scrambled segment back.
        for (i, &value) in segment.iter().enumerate() {
            individual[start + i] = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{RandomGenerator, TestDummyRng};
    use ndarray::{Array1, array};
    use rand::RngCore;
    use rstest::rstest;

    /// A fake RandomGenerator for scramble mutation testing.
    struct FakeRandomGeneratorScramble {
        // Predefined values for segment boundary selection.
        usize_values: Vec<usize>,
        fake_rng: TestDummyRng,
    }

    impl FakeRandomGeneratorScramble {
        fn new(usize_values: Vec<usize>) -> Self {
            Self {
                usize_values,
                fake_rng: TestDummyRng,
            }
        }
    }

    impl RandomGenerator for FakeRandomGeneratorScramble {
        fn rng(&mut self) -> &mut dyn RngCore {
            &mut self.fake_rng
        }

        fn gen_range_usize(&mut self, _min: usize, _max: usize) -> usize {
            self.usize_values.remove(0)
        }

        fn shuffle_vec(&mut self, vector: &mut Vec<f64>) {
            let vector_clone = vector.clone();
            vector[0] = vector_clone[1];
            vector[1] = vector_clone[0];
            vector[2] = vector_clone[3];
            vector[3] = vector_clone[2];
        }
    }

    #[rstest(rng_boundaries,
        case(vec![0, 4]),
        case(vec![4, 0])
    )]
    fn test_scramble_mutation(rng_boundaries: Vec<usize>) {
        // Test individual: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        // Note that selected indexes are 0 and 4 so the segment will be: [0.0, 1.0, 2.0, 3.0]
        // the shuffle mock is simple, it's only  0.0 <-> 1.0 and 2.0 <-> 3.0
        // So the expected individual is [1.0, 0.0, 3.0, 2.0, 4.0, 5.0]
        let mut individual: Array1<f64> = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut rng = FakeRandomGeneratorScramble::new(rng_boundaries);
        let mutation_operator = ScrambleMutation::new();
        assert_eq!(mutation_operator.name(), "ScrambleMutation");
        {
            let view = individual.view_mut();
            mutation_operator.mutate(view, &mut rng);
        }
        let expected: Array1<f64> = array![1.0, 0.0, 3.0, 2.0, 4.0, 5.0];
        assert_eq!(individual, expected);
    }

    #[test]
    fn test_scramble_mutation_same_idx() {
        // in this test gen range return 0 twice, so no segment..
        let mut individual: Array1<f64> = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut rng = FakeRandomGeneratorScramble::new(vec![0, 0]);
        let mutation_operator = ScrambleMutation::new();
        {
            let view = individual.view_mut();
            mutation_operator.mutate(view, &mut rng);
        }
        let expected: Array1<f64> = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(individual, expected);
    }
}
