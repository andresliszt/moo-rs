use std::fmt;
use std::fmt::Debug;

use crate::{
    duplicates::PopulationCleaner,
    genetic::{Population, PopulationGenes},
    operators::{CrossoverOperator, MutationOperator, SelectionOperator},
    random::RandomGenerator,
};

#[derive(Debug)]
pub struct Evolve<Sel, Cross, Mut, DC>
where
    Sel: SelectionOperator,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    DC: PopulationCleaner,
{
    selection: Sel,
    crossover: Cross,
    mutation: Mut,
    pub duplicates_cleaner: Option<DC>,
    mutation_rate: f64,
    crossover_rate: f64,
    lower_bound: Option<f64>,
    upper_bound: Option<f64>,
}

#[derive(Debug)]
pub enum EvolveError {
    EmptyMatingResult {
        message: String,
        current_offspring_count: usize,
        required_offsprings: usize,
    },
}

impl fmt::Display for EvolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvolveError::EmptyMatingResult {
                message,
                current_offspring_count,
                required_offsprings,
            } => {
                write!(
                    f,
                    "{} (generated offsprings: {}, required: {})",
                    message, current_offspring_count, required_offsprings
                )
            }
        }
    }
}

impl<Sel, Cross, Mut, DC> Evolve<Sel, Cross, Mut, DC>
where
    Sel: SelectionOperator,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    DC: PopulationCleaner,
{
    pub fn new(
        selection: Sel,
        crossover: Cross,
        mutation: Mut,
        duplicates_cleaner: Option<DC>,
        mutation_rate: f64,
        crossover_rate: f64,
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
    ) -> Self {
        Self {
            selection,
            crossover,
            mutation,
            duplicates_cleaner,
            mutation_rate,
            crossover_rate,
            lower_bound,
            upper_bound,
        }
    }

    /// Performs a single-step crossover + mutation for a batch of selected parents.
    ///
    /// Before returning the offsprings (PopulationGenes Array2), it clamps each gene
    /// to the specified lower and upper bounds (if provided).
    fn mating_batch(
        &self,
        parents_a: &PopulationGenes,
        parents_b: &PopulationGenes,
        rng: &mut dyn RandomGenerator,
    ) -> PopulationGenes {
        // 1) Perform crossover in one batch.
        let mut offsprings = self
            .crossover
            .operate(parents_a, parents_b, self.crossover_rate, rng);
        // 2) Perform mutation in one batch (often in-place).
        self.mutation
            .operate(&mut offsprings, self.mutation_rate, rng);
        // Clamp each gene's value if bounds are provided.
        if let Some(lb) = self.lower_bound {
            for x in offsprings.iter_mut() {
                *x = (*x).max(lb);
            }
        }
        if let Some(ub) = self.upper_bound {
            for x in offsprings.iter_mut() {
                *x = (*x).min(ub);
            }
        }
        offsprings
    }

    /// Cleans duplicates in `genes`, optionally comparing against a reference population.
    /// If no duplicates_cleaner is provided, returns the genes unchanged.
    pub fn clean_duplicates(
        &self,
        genes: PopulationGenes,
        reference: Option<&PopulationGenes>,
    ) -> PopulationGenes {
        if let Some(ref cleaner) = self.duplicates_cleaner {
            cleaner.remove(&genes, reference)
        } else {
            genes
        }
    }

    /// Generates up to `n_offsprings` unique offspring in multiple iterations (up to `max_iter`).
    ///
    /// The logic is as follows:
    /// 1) Accumulate offspring rows in a Vec<Vec<f64>>.
    /// 2) In each iteration, generate a new batch of offspring via mating_batch.
    /// 3) Clean duplicates within the new offspring.
    /// 4) Clean duplicates between the new offspring and the current population.
    /// 5) Clean duplicates between the new offspring and the already accumulated offspring.
    /// 6) Append the new unique offspring to the accumulator.
    /// 7) Repeat until the desired number is reached.
    pub fn evolve(
        &self,
        population: &Population,
        n_offsprings: usize,
        max_iter: usize,
        rng: &mut dyn RandomGenerator,
    ) -> Result<PopulationGenes, EvolveError> {
        // Accumulate offspring rows in a Vec<Vec<f64>>
        let mut all_offsprings: Vec<Vec<f64>> = Vec::with_capacity(n_offsprings);
        let num_genes = population.genes.ncols();
        let mut iterations = 0;

        while all_offsprings.len() < n_offsprings && iterations < max_iter {
            let remaining = n_offsprings - all_offsprings.len();
            // NOTE: Currently, pymoors implements 2-parent crossover producing 2 children.
            let crossover_needed = remaining / 2 + 1;
            let (parents_a, parents_b) = self.selection.operate(population, crossover_needed, rng);

            // Create offspring from these parents (crossover + mutation)
            let mut new_offsprings = self.mating_batch(&parents_a.genes, &parents_b.genes, rng);
            // Clean duplicates within the new offspring (internal cleaning)
            new_offsprings = self.clean_duplicates(new_offsprings, None);
            // Clean duplicates between new offspring and the current population.
            new_offsprings = self.clean_duplicates(new_offsprings, Some(&population.genes));
            // If we have already accumulated offspring, clean new offspring against them.
            if !all_offsprings.is_empty() {
                let acc_array = PopulationGenes::from_shape_vec(
                    (all_offsprings.len(), num_genes),
                    all_offsprings.iter().flatten().cloned().collect(),
                )
                .expect("Failed to create accumulator array");
                new_offsprings = self.clean_duplicates(new_offsprings, Some(&acc_array));
            }
            // Append the new unique offspring to the accumulator.
            for row in new_offsprings.outer_iter() {
                if all_offsprings.len() >= n_offsprings {
                    break;
                }
                all_offsprings.push(row.to_vec());
            }
            iterations += 1;
        }

        if all_offsprings.is_empty() {
            return Err(EvolveError::EmptyMatingResult {
                message: "No offspring were generated.".to_string(),
                current_offspring_count: 0,
                required_offsprings: n_offsprings,
            });
        }

        // Convert Vec<Vec<f64>> into a single Array2.
        let all_offsprings_len = all_offsprings.len();
        let offspring_data: Vec<f64> = all_offsprings.into_iter().flatten().collect();
        let offspring_array =
            PopulationGenes::from_shape_vec((all_offsprings_len, num_genes), offspring_data)
                .expect("Failed to create offspring array from the accumulated data");

        if offspring_array.nrows() < n_offsprings {
            println!(
                "Warning: Only {} offspring were generated out of the desired {}.",
                offspring_array.nrows(),
                n_offsprings
            );
        }

        Ok(offspring_array)
    }
}
