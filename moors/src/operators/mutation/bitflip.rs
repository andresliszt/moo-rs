use ndarray::ArrayViewMut1;

use crate::{operators::MutationOperator, random::RandomGenerator};

#[derive(Debug, Clone)]
/// Mutation operator that flips bits in a binary individual with a specified mutation rate
pub struct BitFlipMutation {
    pub gene_mutation_rate: f64,
}

impl BitFlipMutation {
    pub fn new(gene_mutation_rate: f64) -> Self {
        Self { gene_mutation_rate }
    }
}

impl MutationOperator for BitFlipMutation {
    fn mutate<'a>(&self, mut individual: ArrayViewMut1<'a, f64>, rng: &mut impl RandomGenerator) {
        for gene in individual.iter_mut() {
            if rng.gen_bool(self.gene_mutation_rate) {
                *gene = if *gene == 0.0 { 1.0 } else { 0.0 };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{RandomGenerator, TestDummyRng};
    use ndarray::array;

    /// A fake RandomGenerator for testing that always returns `true` for `gen_bool`.
    struct FakeRandomGeneratorTrue {
        dummy: TestDummyRng,
    }

    impl FakeRandomGeneratorTrue {
        fn new() -> Self {
            Self {
                dummy: TestDummyRng,
            }
        }
    }

    impl RandomGenerator for FakeRandomGeneratorTrue {
        type R = TestDummyRng;
        fn rng(&mut self) -> &mut TestDummyRng {
            &mut self.dummy
        }
        fn gen_bool(&mut self, _p: f64) -> bool {
            // Always return true so that every gene is mutated.
            true
        }
    }

    #[test]
    fn test_bit_flip_mutation_controlled() {
        // Create a population with two individuals:
        // - The first individual is all zeros.
        // - The second individual is all ones.
        let mut pop = array![[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]];

        // Create a BitFlipMutation operator with a gene mutation rate of 1.0,
        // so every gene should be considered for mutation.
        let mutation_operator = BitFlipMutation::new(1.0);
        // Use our controlled fake RNG which always returns true for gen_bool.
        let mut rng = FakeRandomGeneratorTrue::new();

        // Mutate the population. The `operate` method (from MutationOperator) should
        // call `mutate` on each individual.
        mutation_operator.operate(&mut pop, 1.0, &mut rng);

        // After mutation, every bit should be flipped:
        // - The first individual (originally all 0.0) becomes all 1.0.
        // - The second individual (originally all 1.0) becomes all 0.0.
        let expected_pop = array![[1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]];
        assert_eq!(expected_pop, pop);
    }
}
