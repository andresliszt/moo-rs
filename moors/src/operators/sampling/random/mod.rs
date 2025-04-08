mod binary;
mod float;
mod int;

pub use binary::RandomSamplingBinary;
pub use float::RandomSamplingFloat;
pub use int::RandomSamplingInt;

#[cfg(test)]
mod tests {
    use crate::random::{RandomGenerator, TestDummyRng};
    use rand::RngCore;

    use crate::operators::sampling::random::binary::RandomSamplingBinary;
    use crate::operators::sampling::random::float::RandomSamplingFloat;
    use crate::operators::sampling::random::int::RandomSamplingInt;
    use crate::operators::{GeneticOperator, SamplingOperator};

    /// A controlled fake RandomGenerator for testing purposes.
    /// It returns predictable values:
    /// - `gen_range_f64(min, _max)` always returns `min`
    /// - `gen_bool(_p)` always returns `false`
    struct FakeRandomGenerator {
        dummy: TestDummyRng,
    }

    impl FakeRandomGenerator {
        fn new() -> Self {
            Self {
                dummy: TestDummyRng,
            }
        }
    }

    impl RandomGenerator for FakeRandomGenerator {
        fn rng(&mut self) -> &mut dyn RngCore {
            &mut self.dummy
        }
        fn gen_range_usize(&mut self, min: usize, _max: usize) -> usize {
            min
        }
        fn gen_range_f64(&mut self, min: f64, _max: f64) -> f64 {
            min
        }
        fn gen_usize(&mut self) -> usize {
            0
        }
        fn gen_bool(&mut self, _p: f64) -> bool {
            false
        }
    }

    #[test]
    fn test_random_sampling_float_controlled() {
        let sampler = RandomSamplingFloat::new(-1.0, 1.0);
        assert_eq!(sampler.name(), "RandomSamplingFloat");
        let mut rng = FakeRandomGenerator::new();

        // Generate a population of 10 individuals, each with 5 genes.
        let population = sampler.operate(10, 5, &mut rng);

        // Since our fake returns the minimum for every call to `gen_range_f64`,
        // every gene in the population should be -1.0.
        for gene in population.iter() {
            assert_eq!(*gene, -1.0);
        }
    }

    #[test]
    fn test_random_sampling_int_controlled() {
        let sampler = RandomSamplingInt::new(0, 10);
        assert_eq!(sampler.name(), "RandomSamplingInt");
        let mut rng = FakeRandomGenerator::new();

        let population = sampler.operate(10, 5, &mut rng);

        // The operator uses `gen_range_f64` (with `min` as 0.0) for each gene,
        // so every gene should be 0.0.
        for gene in population.iter() {
            assert_eq!(*gene, 0.0);
        }
    }

    #[test]
    fn test_random_sampling_binary_controlled() {
        let sampler = RandomSamplingBinary::new();
        assert_eq!(sampler.name(), "RandomSamplingBinary");
        let mut rng = FakeRandomGenerator::new();

        let population = sampler.operate(10, 5, &mut rng);

        // Since our fake returns false for every call to `gen_bool(0.5)`,
        // each gene will be 0.0 (because the sampling operator maps false to 0.0).
        for gene in population.iter() {
            assert_eq!(*gene, 0.0);
        }
    }
}
