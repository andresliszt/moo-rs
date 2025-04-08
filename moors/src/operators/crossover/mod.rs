use crate::genetic::{IndividualGenes, PopulationGenes};
use crate::operators::GeneticOperator;
use crate::random::RandomGenerator;

pub mod exponential;
pub mod order;
pub mod sbx;
pub mod single_point;
pub mod uniform_binary;

pub trait CrossoverOperator: GeneticOperator {
    fn n_offsprings_per_crossover(&self) -> usize {
        2
    }

    /// Performs crossover between two parents to produce two offspring.
    fn crossover(
        &self,
        parent_a: &IndividualGenes,
        parent_b: &IndividualGenes,
        rng: &mut dyn RandomGenerator,
    ) -> (IndividualGenes, IndividualGenes);

    /// Applies the crossover operator to the population.
    /// Takes two parent populations and returns two offspring populations.
    /// Includes a `crossover_rate` to determine which pairs undergo crossover.
    fn operate(
        &self,
        parents_a: &PopulationGenes,
        parents_b: &PopulationGenes,
        crossover_rate: f64,
        rng: &mut dyn RandomGenerator,
    ) -> PopulationGenes {
        let population_size = parents_a.nrows();
        assert_eq!(
            population_size,
            parents_b.nrows(),
            "Parent populations must be of the same size"
        );

        let num_genes = parents_a.ncols();
        assert_eq!(
            num_genes,
            parents_b.ncols(),
            "Parent individuals must have the same number of genes"
        );

        // Prepare flat vectors to collect offspring genes
        let mut flat_offspring =
            Vec::with_capacity(self.n_offsprings_per_crossover() * population_size * num_genes);

        for i in 0..population_size {
            let parent_a = parents_a.row(i).to_owned();
            let parent_b = parents_b.row(i).to_owned();

            if rng.gen_proability() <= crossover_rate {
                // Perform crossover
                let (child_a, child_b) = self.crossover(&parent_a, &parent_b, rng);
                flat_offspring.extend(child_a.into_iter());
                flat_offspring.extend(child_b.into_iter());
            } else {
                // Keep parents as offspring
                flat_offspring.extend(parent_a.into_iter());
                flat_offspring.extend(parent_b.into_iter());
            }
        }

        // Create PopulationGenes directly from the flat vectors
        let offspring_population = PopulationGenes::from_shape_vec(
            (
                self.n_offsprings_per_crossover() * population_size,
                num_genes,
            ),
            flat_offspring,
        )
        .expect("Failed to create offspring population");
        offspring_population
    }
}
