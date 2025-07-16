use ndarray::ArrayViewMut1;

use crate::{operators::MutationOperator, random::RandomGenerator};

#[derive(Debug, Clone)]
/// Mutation operator that inverts a contiguous segment of a binary individual.
pub struct InversionMutation;

impl MutationOperator for InversionMutation {
    fn mutate<'a>(&self, mut individual: ArrayViewMut1<'a, f64>, rng: &mut impl RandomGenerator) {
        let len = individual.len();
        // Select two random indices within the chromosome
        let mut start = rng.gen_range_usize(0, len);
        let mut end = rng.gen_range_usize(0, len);
        // Ensure start <= end
        if start > end {
            std::mem::swap(&mut start, &mut end);
        }
        // Reverse the segment between start and end, inclusive
        for k in 0..(end - start).div_ceil(2) {
            let i = start + k;
            let j = end - k;
            let tmp = individual[i];
            individual[i] = individual[j];
            individual[j] = tmp;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{RandomGenerator, TestDummyRng};
    use ndarray::array;

    /// A fake RNG that selects a fixed segment [1, 4] for inversion.
    struct FakeRandomGeneratorInversion {
        dummy: TestDummyRng,
        counter: usize,
    }

    impl FakeRandomGeneratorInversion {
        fn new() -> Self {
            Self {
                dummy: TestDummyRng,
                counter: 0,
            }
        }
    }

    impl RandomGenerator for FakeRandomGeneratorInversion {
        type R = TestDummyRng;

        fn rng(&mut self) -> &mut TestDummyRng {
            &mut self.dummy
        }

        // Always return deterministic indices: first call -> 1, second -> 4
        fn gen_range_usize(&mut self, low: usize, high: usize) -> usize {
            let result = if self.counter == 0 { low + 1 } else { high - 2 };
            self.counter += 1;
            result
        }
    }

    #[test]
    fn test_inversion_mutation_controlled() {
        // Single binary individual: [0, 1, 1, 0, 0, 1]
        let mut pop = array![[0.0, 1.0, 1.0, 0.0, 0.0, 1.0]];

        let mutation_operator = InversionMutation;
        let mut rng = FakeRandomGeneratorInversion::new();

        // Directly call mutate on the individual view
        mutation_operator.mutate(pop.row_mut(0), &mut rng);

        // Original segment [1, 1, 0, 0] at positions [1..4] is reversed to [0, 0, 1, 1]
        let expected = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        assert_eq!(expected, pop.row(0));
    }
}
