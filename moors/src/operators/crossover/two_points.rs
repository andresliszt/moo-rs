use crate::operators::CrossoverOperator;
use crate::random::RandomGenerator;
use ndarray::{Array1, Axis, concatenate, s};

#[derive(Debug, Clone)]
/// Two-point crossover operator for binary-encoded individuals.
pub struct TwoPointBinaryCrossover;

impl CrossoverOperator for TwoPointBinaryCrossover {
    fn crossover(
        &self,
        parent_a: &Array1<f64>,
        parent_b: &Array1<f64>,
        rng: &mut impl RandomGenerator,
    ) -> (Array1<f64>, Array1<f64>) {
        let len = parent_a.len();
        assert_eq!(len, parent_b.len(), "Parents must have the same length");
        if len < 2 {
            return (parent_a.clone(), parent_b.clone());
        }
        // pick two distinct points between 1 and len-1
        let mut c1 = rng.gen_range_usize(1, len);
        let mut c2 = rng.gen_range_usize(1, len);
        if c1 > c2 {
            std::mem::swap(&mut c1, &mut c2);
        } else if c1 == c2 {
            // ensure different: move c2 forward if possible, otherwise backward
            if c2 < len - 1 {
                c2 += 1;
            } else {
                c1 -= 1;
            }
        }
        // build offspring: segments [0..c1), [c1..c2), [c2..len)
        let a0 = parent_a.slice(s![..c1]);
        let a1 = parent_a.slice(s![c1..c2]);
        let a2 = parent_a.slice(s![c2..]);
        let b0 = parent_b.slice(s![..c1]);
        let b1 = parent_b.slice(s![c1..c2]);
        let b2 = parent_b.slice(s![c2..]);

        let offspring_a = concatenate![Axis(0), a0, b1, a2];
        let offspring_b = concatenate![Axis(0), b0, a1, b2];
        (offspring_a, offspring_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{RandomGenerator, TestDummyRng};
    use ndarray::array;

    /// Fake RNG controlling two cut points.
    struct FakeTwoPointRng {
        cuts: Vec<usize>,
        idx: usize,
        dummy: TestDummyRng,
    }
    impl FakeTwoPointRng {
        fn new(cuts: Vec<usize>) -> Self {
            Self {
                cuts,
                idx: 0,
                dummy: TestDummyRng,
            }
        }
    }
    impl RandomGenerator for FakeTwoPointRng {
        type R = TestDummyRng;
        fn rng(&mut self) -> &mut TestDummyRng {
            &mut self.dummy
        }
        fn gen_range_usize(&mut self, _min: usize, _max: usize) -> usize {
            let val = self.cuts[self.idx];
            self.idx += 1;
            val
        }
    }

    #[test]
    fn test_two_point_binary_crossover_controlled() {
        let parent_a = array![0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let parent_b = array![1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        // Fake RNG will pick cut points 2 and 4
        let mut rng = FakeTwoPointRng::new(vec![2, 4]);
        let op = TwoPointBinaryCrossover;
        let (child_a, child_b) = op.crossover(&parent_a, &parent_b, &mut rng);
        // segments: [0..2], [2..4], [4..]
        // child_a = A[0..2] + B[2..4] + A[4..] => [0,1] + [0,1] + [1,0] = [0,1,0,1,1,0]
        // child_b = B[0..2] + A[2..4] + B[4..] => [1,0] + [1,0] + [0,1] = [1,0,1,0,0,1]
        assert_eq!(child_a, array![0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
        assert_eq!(child_b, array![1.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
    }
}
