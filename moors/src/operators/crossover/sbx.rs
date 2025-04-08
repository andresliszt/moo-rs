use ndarray::Array1;

use crate::genetic::IndividualGenes;
use crate::operators::{CrossoverOperator, GeneticOperator};
use crate::random::RandomGenerator;

/// Simulated Binary Crossover (SBX) operator for real-coded genetic algorithms.
///
/// # SBX Overview
///
/// - SBX mimics the behavior of single-point crossover in binary-coded GAs,
///   but for continuous variables.
/// - The parameter `distribution_index` (η) controls how far offspring can deviate
///   from the parents. Larger η values produce offspring closer to the parents (less exploration),
///   while smaller values allow offspring to be more spread out (more exploration).
///
/// **Reference**: Deb, Kalyanmoy, and R. B. Agrawal. "Simulated binary crossover for continuous search space."
///              Complex systems 9.2 (1995): 115-148.
#[derive(Clone, Debug)]
pub struct SimulatedBinaryCrossover {
    /// Distribution index (η) that controls offspring spread.
    pub distribution_index: f64,
}

impl SimulatedBinaryCrossover {
    /// Creates a new `SimulatedBinaryCrossover` operator with the given distribution index
    pub fn new(distribution_index: f64) -> Self {
        Self { distribution_index }
    }
}

/// Performs SBX crossover on two parent solutions represented as Array1<f64>.
///
/// For each gene, if the two parent values differ sufficiently, the SBX operator is applied.
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
    prob_exchange: f64,
    rng: &mut dyn RandomGenerator,
) -> (Array1<f64>, Array1<f64>) {
    let n = p1.len();
    let eps = 1e-16;
    let mut offspring1 = p1.clone();
    let mut offspring2 = p2.clone();

    for i in 0..n {
        let gene1 = p1[i];
        let gene2 = p2[i];

        // Skip if the genes are nearly identical or if there is no valid range.
        if (gene1 - gene2).abs() < eps {
            continue;
        }

        // Generate random numbers for beta_q computation and for exchange decision.
        let r_beta = rng.gen_proability();
        let r_exchange = rng.gen_proability();

        // Order the gene values so that y1 is the smaller and y2 is the larger.
        let (y1, y2) = if gene1 < gene2 {
            (gene1, gene2)
        } else {
            (gene2, gene1)
        };
        let delta = y2 - y1;

        // Compute beta_q using Deb & Agrawal's method.
        let beta_q = if r_beta <= 0.5 {
            (2.0 * r_beta).powf(1.0 / (distribution_index + 1.0))
        } else {
            (1.0 / (2.0 * (1.0 - r_beta))).powf(1.0 / (distribution_index + 1.0))
        };

        // Compute offspring gene values.
        let c1 = 0.5 * ((y1 + y2) - beta_q * delta);
        let c2 = 0.5 * ((y1 + y2) + beta_q * delta);

        // With probability `prob_exchange`, swap the offspring values.
        let (new1, new2) = if r_exchange < prob_exchange {
            (c2, c1)
        } else {
            (c1, c2)
        };
        offspring1[i] = new1;
        offspring2[i] = new2;
    }

    (offspring1, offspring2)
}

impl GeneticOperator for SimulatedBinaryCrossover {
    fn name(&self) -> String {
        format!(
            "SimulatedBinaryCrossover(distribution_index={})",
            self.distribution_index
        )
    }
}

impl CrossoverOperator for SimulatedBinaryCrossover {
    fn crossover(
        &self,
        parent_a: &IndividualGenes,
        parent_b: &IndividualGenes,
        rng: &mut dyn RandomGenerator,
    ) -> (IndividualGenes, IndividualGenes) {
        // TODO: Enable prob_exchange
        sbx_crossover_array(parent_a, parent_b, self.distribution_index, 0.0, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genetic::IndividualGenes;
    use crate::random::{RandomGenerator, TestDummyRng};
    use ndarray::array;

    /// A fake random generator for controlled testing of SBX.
    /// It provides predetermined probability values via `gen_proability`.
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
        fn rng(&mut self) -> &mut dyn rand::RngCore {
            &mut self.dummy
        }
        fn gen_proability(&mut self) -> f64 {
            self.probability_values.remove(0)
        }
    }

    #[test]
    fn test_simulated_binary_crossover() {
        // Define two parent genes as IndividualGenes.
        // For gene 0: p1 = 1.0, p2 = 3.0 (SBX is applied).
        // For gene 1: p1 = 5.0, p2 = 5.0 (no crossover is applied).
        let parent_a: IndividualGenes = array![1.0, 5.0];
        let parent_b: IndividualGenes = array![3.0, 5.0];

        // Create the SBX operator with distribution_index = 2.0.
        let operator = SimulatedBinaryCrossover::new(2.0);

        // For gene 0, we need to supply:
        // - A value for r_beta (e.g., 0.25) and a value for r_exchange (e.g., 0.9).
        // For gene 1, no SBX is applied because the genes are identical.
        let mut fake_rng = FakeRandom::new(vec![
            0.25, // r_beta for gene 0.
            0.9,  // r_exchange for gene 0.
                  // No random values for gene 1.
        ]);

        let (child_a, child_b) = operator.crossover(&parent_a, &parent_b, &mut fake_rng);
        let tol = 1e-8;

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
            (child_a[0] - 1.2062994741).abs() < tol,
            "Gene 0 of child_a not as expected"
        );
        assert!(
            (child_b[0] - 2.7937005259).abs() < tol,
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
