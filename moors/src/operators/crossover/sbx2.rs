use crate::{CrossoverOperator, RandomGenerator};
use ndarray::Array1;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct SBXCrossover {
    /// Distribution index (η) that controls offspring spread.
    pub distribution_index: f64,
    /// one (min,max) per variable
    pub ranges: Arc<Vec<(f64, f64)>>,
    pub swap_prob: f64,
}

impl SBXCrossover {
    /// Creates a new `SBXCrossover` operator with the given distribution index and no swap
    pub fn new(distribution_index: f64, ranges: Arc<Vec<(f64, f64)>>) -> Self {
        Self {
            distribution_index,
            ranges,
            swap_prob: 0.0,
        }
    }
}

/// Performs SBX crossover on two parent solutions represented as Array1<f64>.
///
/// For each gene, if the two parent values differ sufficiently, the SBX asymmetric operator is applied.
///
/// # Arguments
///
/// * `p1` - Parent 1 genes.
/// * `p2` - Parent 2 genes.
/// * `distribution_index` - SBX distribution index (η).
/// * `prob_exchange` - Probability to swap the offspring values.
/// * `rng` - A mutable random number generator.
///
/// # Returns
///
/// A tuple containing two offspring as Array1<f64>.
pub fn sbx_crossover_array(
    p1: &Array1<f64>,
    p2: &Array1<f64>,
    distribution_index: f64,
    swap_prob: f64,
    rng: &mut impl RandomGenerator,
    ranges: &[(f64, f64)],
) -> (Array1<f64>, Array1<f64>) {
    let n = p1.len();
    assert_eq!(n, p2.len(), "parents must be same length");
    assert_eq!(n, ranges.len(), "ranges must match gene length");

    let mut off1 = p1.clone();
    let mut off2 = p2.clone();
    let eps = 1e-10;

    for i in 0..n {
        let x1 = p1[i];
        let x2 = p2[i];
        // SBX only if value differ and a “coin‐flip” allows it
        if (x1 - x2).abs() > eps && rng.gen_probability() < 0.5 {
            let (lb, ub) = ranges[i];

            // order them
            let (y1, y2) = if x1 < x2 { (x1, x2) } else { (x2, x1) };
            let delta = y2 - y1;
            let rand = rng.gen_probability();

            // —— child #1 (lower‐bound aware) ——
            // β = 1 / (1 + 2*(y1−lb)/Δ)
            let beta = 1.0 / (1.0 + 2.0 * (y1 - lb) / delta);
            let alpha = 2.0 - beta.powf(distribution_index + 1.0);

            let betaq = if rand <= 1.0 / alpha {
                (rand * alpha).powf(1.0 / (distribution_index + 1.0))
            } else {
                (1.0 / (2.0 - rand * alpha)).powf(1.0 / (distribution_index + 1.0))
            };

            let mut c1 = 0.5 * ((y1 + y2) - betaq * delta);

            // —— child #2 (upper‐bound aware) ——
            let beta = 1.0 / (1.0 + 2.0 * (ub - y2) / delta);
            let alpha = 2.0 - beta.powf(distribution_index + 1.0);

            let betaq = if rand <= 1.0 / alpha {
                (rand * alpha).powf(1.0 / (distribution_index + 1.0))
            } else {
                (1.0 / (2.0 - rand * alpha)).powf(1.0 / (distribution_index + 1.0))
            };

            let mut c2 = 0.5 * ((y1 + y2) + betaq * delta);

            // clamp both to [lb,ub]
            c1 = c1.clamp(lb, ub);
            c2 = c2.clamp(lb, ub);

            // possibly swapping them
            if swap_prob > 0.0 && rng.gen_probability() < swap_prob {
                off1[i] = c2;
                off2[i] = c1;
            } else {
                off1[i] = c1;
                off2[i] = c2;
            }
        }
        // else leave off1[i]=x1, off2[i]=x2
    }

    (off1, off2)
}

impl CrossoverOperator for SBXCrossover {
    fn crossover(
        &self,
        parent_a: &Array1<f64>,
        parent_b: &Array1<f64>,
        rng: &mut impl RandomGenerator,
    ) -> (Array1<f64>, Array1<f64>) {
        sbx_crossover_array(
            parent_a,
            parent_b,
            self.distribution_index,
            self.swap_prob,
            rng,
            self.ranges.as_slice(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{RandomGenerator, TestDummyRng};
    use ndarray::array;

    /// A fake random generator for controlled testing of SBX.
    /// It provides predetermined probability values via `gen_probability`.
    struct FakeRandom {
        /// Predefined probability values to be returned sequentially.
        probability_values: Vec<f64>,
        /// Dummy RNG to satisfy the trait requirement.
        dummy: TestDummyRng,
    }

    impl FakeRandom {
        /// Creates a new instance with the given probability responses.
        /// For each gene where SBX is applied, two values are used:
        /// one for r_beta and one for r_exchange.
        fn new(probability_values: Vec<f64>) -> Self {
            Self {
                probability_values,
                dummy: TestDummyRng,
            }
        }
    }

    impl RandomGenerator for FakeRandom {
        type R = TestDummyRng;
        fn rng(&mut self) -> &mut TestDummyRng {
            &mut self.dummy
        }
        fn gen_probability(&mut self) -> f64 {
            self.probability_values[0]
        }
    }

    #[test]
    fn test_sbxcrossover() {
        let var_ranges = Arc::new(vec![(0.0, 4.0), (4.0, 6.0)]);
        // Define two parent genes as IndividualGenes.
        // For gene 0 SBX is applied
        // For gene 1 no crossover is applied
        let parent_a = array![1.0, 5.0];
        let parent_b = array![3.0, 5.0];
        // Create the SBX operator
        let operator = SBXCrossover::new(2.0, var_ranges.clone());
        // For gene 0, we need to supply:
        // - A value for r_beta (e.g., 0.25) and a value for r_exchange (e.g., 0.9).
        // For gene 1, no SBX is applied because the genes are identical.
        let mut fake_rng = FakeRandom::new(vec![
            0.25, // r_beta for gene 0.
                 // No random values for gene 1.
        ]);

        let (child_a, child_b) = operator.crossover(&parent_a, &parent_b, &mut fake_rng);
        let tol = 1e-3;

        println!("After crossover: {:?}", (child_a.clone(), child_b.clone()));

        // For gene 0:
        // With distribution_index = 2.0 and r_beta = 0.25:
        //   beta_q = (2 * 0.25)^(1/(2+1)) = 0.5^(1/3) ≈ 0.7937005259.
        // Then:
        //   c1 = 0.5*((1.0+3.0) - 0.7937005259*(3.0-1.0))
        //      ≈ 0.5*(4.0 - 1.5874010518)
        //      ≈ 1.2062994741.
        //   c2 = 0.5*((1.0+3.0) + 0.7937005259*(3.0-1.0))
        //      ≈ 0.5*(4.0 + 1.5874010518)
        //      ≈ 2.7937005259.
        // For gene 1, since the genes are identical, no crossover is applied.
        assert!(
            (child_a[0] - 1.223).abs() < tol,
            "Gene 0 of child_a not as expected"
        );
        assert!(
            (child_b[0] - 2.776).abs() < tol,
            "Gene 0 of child_b not as expected"
        );
        assert!(
            (child_a[1] - 5.0).abs() < tol,
            "Gene 1 of child_a not as expected"
        );
        assert!(
            (child_b[1] - 5.0).abs() < tol,
            "Gene 1 of child_b not as expected"
        );
    }
}
