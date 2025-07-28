use crate::{MutationOperator, RandomGenerator};
use ndarray::ArrayViewMut1;
use std::sync::Arc;

/// Polynomial mutation (Deb’s PM) with per-gene bounds.
#[derive(Clone)]
pub struct PolynomialMutation {
    /// per-gene mutation chance pₘ
    pub gene_mutation_rate: f64,
    /// distribution index ηₘ
    pub distribution_index: f64,
    pub var_ranges: Arc<Vec<(f64, f64)>>,
}

impl PolynomialMutation {
    pub fn new(
        gene_mutation_rate: f64,
        distribution_index: f64,
        var_ranges: Arc<Vec<(f64, f64)>>,
    ) -> Self {
        Self {
            gene_mutation_rate,
            distribution_index,
            var_ranges,
        }
    }
}

impl MutationOperator for PolynomialMutation {
    fn mutate<'a>(&self, mut individual: ArrayViewMut1<'a, f64>, rng: &mut impl RandomGenerator) {
        let eta = self.distribution_index;
        // mutate each gene with the given probability
        for (gene, range) in individual.iter_mut().zip(self.var_ranges.iter()) {
            // each gene is mutated with the given probability
            if rng.gen_bool(self.gene_mutation_rate) {
                // gene boundaries
                let lb = range.0;
                let ub = range.1;
                let dx = ub - lb;
                // 1) draw u ∈ [0,1)
                let u = rng.gen_range_f64(0.0, 1.0);
                let x = *gene;
                // 2) compute Δ according to Deb’s PM formula
                let delta = if u < 0.5 {
                    let bl = (x - lb) / dx;
                    let b = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - bl).powf(eta + 1.0);
                    b.powf(1.0 / (eta + 1.0)) - 1.0
                } else {
                    let bu = (ub - x) / dx;
                    let b = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - bu).powf(eta + 1.0);
                    1.0 - b.powf(1.0 / (eta + 1.0))
                };
                // 3) apply mutation
                *gene = x + delta * dx;
                // 4) clamp mutated genes
                *gene = gene.clamp(lb, ub);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::operators::mutation::polynomial::PolynomialMutation;
    use crate::{MOORandomGenerator, MutationOperator};
    use ndarray::array;
    use rand::SeedableRng;
    use rand::prelude::StdRng;
    use std::sync::Arc;

    #[test]
    fn test_pm_all_genes() {
        let var_ranges = Arc::new(vec![(-10.0, 0.0), (0.0, 1.0), (1.0, 10.0)]);
        // Create an individual [0.5, 0.5, 0.5]
        let mut pop = array![[-5.5, 0.5, 7.5]];
        let pop_before_mut = pop.clone();
        // Create mutation operator
        let mutation_operator = PolynomialMutation::new(1.0, 20.0, var_ranges.clone());
        let mut rng = MOORandomGenerator::new(StdRng::seed_from_u64(42));
        // Mutate the population
        mutation_operator.operate(&mut pop, 1.0, &mut rng);
        // mutated_pop should differ from pop, but let's just ensure it's not identical
        assert_ne!(pop, pop_before_mut);
        println!("Original: {:?}", pop_before_mut);
        println!("Mutated: {:?}", pop);
    }
}
