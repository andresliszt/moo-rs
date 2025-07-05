use crate::{operators::CrossoverOperator, random::RandomGenerator};
use ndarray::Array1;

#[derive(Debug, Clone)]
/// Whole arithmetic crossover for real-valued individuals.
/// Samples a single α ∼ U(0,1) and produces two offspring:
///   child1[i] = α * parent_a[i] + (1−α) * parent_b[i]
///   child2[i] = (1−α) * parent_a[i] + α * parent_b[i]
pub struct ArithmeticCrossover;

impl CrossoverOperator for ArithmeticCrossover {
    fn crossover(
        &self,
        parent_a: &Array1<f64>,
        parent_b: &Array1<f64>,
        rng: &mut impl RandomGenerator,
    ) -> (Array1<f64>, Array1<f64>) {
        let len = parent_a.len();
        assert_eq!(len, parent_b.len(), "Parents must have same length");

        // Draw α ∼ Uniform(0,1)
        let alpha = rng.gen_range_f64(0.0, 1.0);

        // Allocate offspring
        let mut child1 = Array1::zeros(len);
        let mut child2 = Array1::zeros(len);

        for i in 0..len {
            let x = parent_a[i];
            let y = parent_b[i];
            child1[i] = alpha * x + (1.0 - alpha) * y;
            child2[i] = (1.0 - alpha) * x + alpha * y;
        }

        (child1, child2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{RandomGenerator, TestDummyRng};
    use ndarray::array;

    /// Fake RNG that returns a fixed α for gen_uniform.
    struct FakeArithmeticRng {
        dummy: TestDummyRng,
        alpha: f64,
    }
    impl FakeArithmeticRng {
        fn new(alpha: f64) -> Self {
            Self {
                dummy: TestDummyRng,
                alpha,
            }
        }
    }
    impl RandomGenerator for FakeArithmeticRng {
        type R = TestDummyRng;
        fn rng(&mut self) -> &mut TestDummyRng {
            &mut self.dummy
        }
        fn gen_range_f64(&mut self, _min: f64, _max: f64) -> f64 {
            self.alpha
        }
        fn gen_bool(&mut self, _p: f64) -> bool {
            false
        }
    }

    #[test]
    fn test_arithmetic_crossover_controlled() {
        let parent_a = array![1.0, 2.0];
        let parent_b = array![3.0, 4.0];
        let mut rng = FakeArithmeticRng::new(0.25);
        let op = ArithmeticCrossover;

        let (child1, child2) = op.crossover(&parent_a, &parent_b, &mut rng);

        // α = 0.25 → child1 = [2.5, 3.5], child2 = [1.5, 2.5]
        assert_eq!(child1, array![2.5, 3.5]);
        assert_eq!(child2, array![1.5, 2.5]);
    }
}
