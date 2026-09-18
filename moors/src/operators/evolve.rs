use derive_builder::Builder;
use ndarray::Array2;
use thiserror::Error;

use std::sync::Arc;

use crate::{
    duplicates::{NoDuplicatesCleaner, PopulationCleaner},
    genetic::{D01, D12, Population},
    operators::repair::{NoRepair, RepairOperator},
    operators::{CrossoverOperator, MutationOperator, SelectionOperator},
    random::RandomGenerator,
};

#[derive(Debug, Clone, Builder)]
#[builder(pattern = "owned")]
pub struct Evolve<Sel, Cross, Mut>
where
    Sel: SelectionOperator,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
{
    selection: Sel,
    crossover: Cross,
    mutation: Mut,
    #[builder(default = "Arc::new(NoDuplicatesCleaner)")]
    pub duplicates_cleaner: Arc<dyn PopulationCleaner>,
    #[builder(default = "Arc::new(NoRepair)")]
    repair: Arc<dyn RepairOperator>,
    mutation_rate: f64,
    crossover_rate: f64,
    lower_bound: Option<f64>,
    upper_bound: Option<f64>,
}

#[derive(Debug, Error)]
pub enum EvolveError {
    #[error("No offsprings were generated in the mating process")]
    EmptyMatingResult,
}

impl<Sel, Cross, Mut> Evolve<Sel, Cross, Mut>
where
    Sel: SelectionOperator,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
{
    /// Performs a single-step crossover + mutation for a batch of selected parents.
    ///
    /// Before returning the offsprings (PopulationGenes Array2), it clamps each gene
    /// to the specified lower and upper bounds (if provided).
    fn mating_batch(
        &self,
        parents_a: &Array2<f64>,
        parents_b: &Array2<f64>,
        rng: &mut impl RandomGenerator,
    ) -> Array2<f64> {
        // 1) Perform crossover in one batch.
        let mut offsprings = self
            .crossover
            .operate(parents_a, parents_b, self.crossover_rate, rng);
        // 2) Perform mutation in one batch (often in-place).
        self.mutation
            .operate(&mut offsprings, self.mutation_rate, rng);
        // 3) Optionally repair infeasible individuals.
        self.repair.operate(&mut offsprings);
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

    /// Overrides the mutation rate used for subsequent matings (e.g. from an
    /// `AdaptiveController`).
    pub fn set_mutation_rate(&mut self, mutation_rate: f64) {
        self.mutation_rate = mutation_rate;
    }

    /// Overrides the crossover rate used for subsequent matings (e.g. from an
    /// `AdaptiveController`).
    pub fn set_crossover_rate(&mut self, crossover_rate: f64) {
        self.crossover_rate = crossover_rate;
    }

    /// Generates up to `num_offsprings` unique offspring in multiple iterations (up to `max_iter`).
    ///
    /// The logic is as follows:
    /// 1) Accumulate offspring rows in a Vec<Vec<f64>>.
    /// 2) In each iteration, generate a new batch of offspring via mating_batch.
    /// 3) Clean duplicates within the new offspring.
    /// 4) Clean duplicates between the new offspring and the current population.
    /// 5) Clean duplicates between the new offspring and the already accumulated offspring.
    /// 6) Append the new unique offspring to the accumulator.
    /// 7) Repeat until the desired number is reached.
    pub fn evolve<ConstrDim>(
        &self,
        population: &Population<Sel::FDim, ConstrDim>,
        num_offsprings: usize,
        max_iter: usize,
        rng: &mut impl RandomGenerator,
    ) -> Result<Array2<f64>, EvolveError>
    where
        ConstrDim: D12,
        <Sel::FDim as ndarray::Dimension>::Smaller: D01,
        <ConstrDim as ndarray::Dimension>::Smaller: D01,
    {
        // Accumulate offspring rows in a Vec<Vec<f64>>
        let mut all_offsprings: Vec<Vec<f64>> = Vec::with_capacity(num_offsprings);
        let num_genes = population.genes.ncols();
        let mut iterations = 0;

        while all_offsprings.len() < num_offsprings && iterations < max_iter {
            let remaining = num_offsprings - all_offsprings.len();
            // NOTE: Currently, pymoors implements 2-parent crossover producing 2 children.
            let crossover_needed = remaining / 2 + 1;
            let (parents_a, parents_b) = self.selection.operate(population, crossover_needed, rng);

            // Create offspring from these parents (crossover + mutation)
            let mut new_offsprings = self.mating_batch(&parents_a.genes, &parents_b.genes, rng);
            // Clean duplicates within the new offspring (internal cleaning)
            new_offsprings = (self.duplicates_cleaner).remove(new_offsprings, None);
            // Clean duplicates between new offspring and the current population.
            new_offsprings =
                (self.duplicates_cleaner).remove(new_offsprings, Some(&population.genes));
            // If we have already accumulated offspring, clean new offspring against them.
            if !all_offsprings.is_empty() {
                let acc_array = Array2::<f64>::from_shape_vec(
                    (all_offsprings.len(), num_genes),
                    all_offsprings.iter().flatten().cloned().collect(),
                )
                .expect("Failed to create accumulator array");
                new_offsprings = (self.duplicates_cleaner).remove(new_offsprings, Some(&acc_array));
            }
            // Append the new unique offspring to the accumulator.
            for row in new_offsprings.outer_iter() {
                if all_offsprings.len() >= num_offsprings {
                    break;
                }
                all_offsprings.push(row.to_vec());
            }
            iterations += 1;
        }

        if all_offsprings.is_empty() {
            return Err(EvolveError::EmptyMatingResult);
        }

        // Convert Vec<Vec<f64>> into a single Array2.
        let all_offsprings_len = all_offsprings.len();
        let offspring_data: Vec<f64> = all_offsprings.into_iter().flatten().collect();
        let offspring_array =
            Array2::<f64>::from_shape_vec((all_offsprings_len, num_genes), offspring_data)
                .expect("Failed to create offspring array from the accumulated data");

        if offspring_array.nrows() < num_offsprings {
            println!(
                "Warning: Only {} offspring were generated out of the desired {}.",
                offspring_array.nrows(),
                num_offsprings
            );
        }

        Ok(offspring_array)
    }
}
