use crate::{
    genetic::IndividualGenesMut,
    operators::{GeneticOperator, MutationOperator},
    random::RandomGenerator,
};

#[derive(Clone, Debug)]
/// Mutation operator that flips bits in a binary individual with a specified mutation rate
pub struct BitFlipMutation {
    pub gene_mutation_rate: f64,
}

impl BitFlipMutation {
    #[allow(dead_code)]
    pub fn new(gene_mutation_rate: f64) -> Self {
        Self { gene_mutation_rate }
    }
}

impl GeneticOperator for BitFlipMutation {
    fn name(&self) -> String {
        "BitFlipMutation".to_string()
    }
}

impl MutationOperator for BitFlipMutation {
    fn mutate<'a>(&self, mut individual: IndividualGenesMut<'a>, rng: &mut dyn RandomGenerator) {
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
    use crate::genetic::PopulationGenes;
    use crate::random::{RandomGenerator, TestDummyRng};
    use ndarray::array;
    use rand::RngCore;

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
        fn rng(&mut self) -> &mut dyn RngCore {
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
        let mut pop: PopulationGenes = array![[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]];

        // Create a BitFlipMutation operator with a gene mutation rate of 1.0,
        // so every gene should be considered for mutation.
        let mutation_operator = BitFlipMutation::new(1.0);
        assert_eq!(mutation_operator.name(), "BitFlipMutation");

        // Use our controlled fake RNG which always returns true for gen_bool.
        let mut rng = FakeRandomGeneratorTrue::new();

        // Mutate the population. The `operate` method (from MutationOperator) should
        // call `mutate` on each individual.
        mutation_operator.operate(&mut pop, 1.0, &mut rng);

        // After mutation, every bit should be flipped:
        // - The first individual (originally all 0.0) becomes all 1.0.
        // - The second individual (originally all 1.0) becomes all 0.0.
        let expected_pop: PopulationGenes =
            array![[1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]];
        assert_eq!(expected_pop, pop);
    }
}
