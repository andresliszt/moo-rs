use crate::operators::{GenesMut, GeneticOperator, MutationOperator};
use crate::random::RandomGenerator;

use rand_distr::{Distribution, Normal};

#[derive(Clone, Debug)]
pub struct GaussianMutation {
    pub gene_mutation_rate: f64,
    pub sigma: f64,
}

impl GaussianMutation {
    pub fn new(gene_mutation_rate: f64, sigma: f64) -> Self {
        Self {
            gene_mutation_rate,
            sigma,
        }
    }
}

impl GeneticOperator for GaussianMutation {
    fn name(&self) -> String {
        "GaussianMutation".to_string()
    }
}

impl MutationOperator for GaussianMutation {
    fn mutate<'a>(&self, mut individual: GenesMut<'a>, rng: &mut dyn RandomGenerator) {
        // Create a normal distribution with mean 0.0 and standard deviation sigma.
        let normal_dist = Normal::new(0.0, self.sigma)
            .expect("Failed to create normal distribution. Sigma must be > 0.");

        // Iterate over each gene in the mutable view.
        for gene in individual.iter_mut() {
            if rng.gen_bool(self.gene_mutation_rate) {
                // Sample a delta from the normal distribution and add it to the gene.
                let delta = normal_dist.sample(rng.rng());
                *gene += delta;
            }
        }
    }
}

impl_py_mutation!(
    "Mutation operator that adds Gaussian noise to float variables.",
    PyGaussianMutation,
    GaussianMutation,
    "GaussianMutation",
    gene_mutation_rate: f64, sigma: f64
);

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::genetic::PopulationGenes;
    use crate::random::MOORandomGenerator;
    use numpy::ndarray::array;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_gaussian_mutation_all_genes() {
        // Create an individual [0.5, 0.5, 0.5]
        let mut pop: PopulationGenes = array![[0.5, 0.5, 0.5]];
        let pop_before_mut = array![[0.5, 0.5, 0.5]];

        // Create operator with 100% chance each gene is mutated, sigma=0.1
        let mutation_operator = GaussianMutation::new(1.0, 0.1);
        assert_eq!(mutation_operator.name(), "GaussianMutation");

        let mut rng = MOORandomGenerator::new(StdRng::seed_from_u64(42));

        // Mutate the population
        mutation_operator.operate(&mut pop, 1.0, &mut rng);

        // mutated_pop should differ from pop, but let's just ensure it's not identical
        assert_ne!(pop, pop_before_mut);

        println!("Original: {:?}", pop_before_mut);
        println!("Mutated: {:?}", pop);
    }

    #[test]
    fn test_gaussian_mutation_no_genes() {
        // If gene_mutation_rate=0.0, no genes are mutated
        let mut pop: PopulationGenes = array![[0.5, 0.5, 0.5]];
        let expected = array![[0.5, 0.5, 0.5]];
        let mutation_operator = GaussianMutation::new(0.0, 0.1);

        let mut rng = MOORandomGenerator::new(StdRng::seed_from_u64(42));
        mutation_operator.operate(&mut pop, 1.0, &mut rng);

        assert_eq!(pop, expected);
    }
}
