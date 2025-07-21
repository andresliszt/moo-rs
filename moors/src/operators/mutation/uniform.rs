use crate::{operators::MutationOperator, random::RandomGenerator};
use ndarray::ArrayViewMut1;

#[derive(Debug, Clone)]
/// Uniform mutation operator that resets each bit to a random 0 or 1
/// with a specified per-gene mutation probability.
pub struct UniformBinaryMutation {
    /// Probability of mutating each gene
    pub gene_mutation_rate: f64,
}

impl UniformBinaryMutation {
    /// Creates a new UniformBinaryMutation with the given per-gene mutation rate.
    pub fn new(gene_mutation_rate: f64) -> Self {
        Self { gene_mutation_rate }
    }
}

impl MutationOperator for UniformBinaryMutation {
    fn mutate<'a>(&self, mut individual: ArrayViewMut1<'a, f64>, rng: &mut impl RandomGenerator) {
        for gene in individual.iter_mut() {
            // With probability gene_mutation_rate, reset this bit...
            if rng.gen_bool(self.gene_mutation_rate) {
                // ...to a fresh random allele (0 or 1)
                *gene = if rng.gen_bool(0.5) { 1.0 } else { 0.0 };
            }
        }
    }
}

/// Mutation operator that resets a real-valued gene
/// to a new U(lower, upper) draw with per-gene rate pₘ.
#[derive(Debug, Clone)]
pub struct UniformRealMutation {
    pub gene_mutation_rate: f64,
    pub lower: f64,
    pub upper: f64,
}

impl UniformRealMutation {
    pub fn new(gene_mutation_rate: f64, lower: f64, upper: f64) -> Self {
        Self {
            gene_mutation_rate,
            lower,
            upper,
        }
    }
}

impl MutationOperator for UniformRealMutation {
    fn mutate<'a>(&self, mut individual: ArrayViewMut1<'a, f64>, rng: &mut impl RandomGenerator) {
        for gene in individual.iter_mut() {
            if rng.gen_bool(self.gene_mutation_rate) {
                // draw uniform in [lower, upper]
                let u: f64 = rng.gen_range_f64(self.lower, self.upper);
                *gene = u;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{RandomGenerator, TestDummyRng};
    use ndarray::array;

    /// Fake RNG that alternates mutate/reset and picks 1 then 0.
    struct FakeUniformRng {
        dummy: TestDummyRng,
        step: usize,
    }

    impl FakeUniformRng {
        fn new() -> Self {
            Self {
                dummy: TestDummyRng,
                step: 0,
            }
        }
    }

    impl RandomGenerator for FakeUniformRng {
        type R = TestDummyRng;
        fn rng(&mut self) -> &mut TestDummyRng {
            &mut self.dummy
        }
        fn gen_bool(&mut self, _p: f64) -> bool {
            // step 0: mutate first gene
            // step 1: draw new allele for first gene → 1
            // step 2: mutate second gene
            // step 3: draw new allele for second gene → 0
            let out = match self.step {
                0 | 2 => true,
                1 => true,  // assign 1.0
                _ => false, // assign 0.0
            };
            self.step += 1;
            out
        }
    }

    #[test]
    fn test_uniform_mutation() {
        // Start with [0, 1]
        let mut pop = array![[0.0, 1.0, 0.0, 0.0]];
        let op = UniformBinaryMutation::new(1.0);
        let mut rng = FakeUniformRng::new();

        op.mutate(pop.row_mut(0), &mut rng);

        // After: first → 1.0, second → 0.0, everything beyond index 1 remains unchanged
        // due to ou fake random generator
        assert_eq!(pop.row(0), array![1.0, 0.0, 0.0, 0.0]);
    }

    /// Fake RNG for real uniform mutation: always mutates and returns predefined values.
    struct FakeRandomGeneratorReal {
        dummy: TestDummyRng,
        values: Vec<f64>,
        idx: usize,
    }

    impl FakeRandomGeneratorReal {
        fn new(values: Vec<f64>) -> Self {
            Self {
                dummy: TestDummyRng,
                values,
                idx: 0,
            }
        }
    }

    impl RandomGenerator for FakeRandomGeneratorReal {
        type R = TestDummyRng;
        fn rng(&mut self) -> &mut TestDummyRng {
            &mut self.dummy
        }
        fn gen_bool(&mut self, _p: f64) -> bool {
            true
        }
        fn gen_range_f64(&mut self, _low: f64, _high: f64) -> f64 {
            let v = self.values[self.idx];
            self.idx += 1;
            v
        }
    }

    #[test]
    fn test_uniform_real_mutation_controlled() {
        // Individual with three real-valued genes
        let mut pop = array![[2.0, -1.0, 0.5]];
        // Mutation rate = 1.0, bounds [-2.0, 3.0]
        let op = UniformRealMutation::new(1.0, -2.0, 3.0);
        // Fake RNG will return 0.0, then 3.0, then -2.0
        let mut rng = FakeRandomGeneratorReal::new(vec![0.0, 3.0, -2.0]);

        op.mutate(pop.row_mut(0), &mut rng);

        // After mutation, genes should match the sequence from the fake RNG
        let expected = array![0.0, 3.0, -2.0];
        assert_eq!(expected, pop.row(0));
    }
}
