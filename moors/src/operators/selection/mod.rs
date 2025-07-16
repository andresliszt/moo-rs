use crate::{
    genetic::{D01, D12, Individual, Population},
    random::RandomGenerator,
};
use ndarray::Dimension;

pub mod moo;
pub mod soo;

// Enum to represent the result of a tournament duel.
#[derive(Debug, PartialEq, Eq)]
pub enum DuelResult {
    LeftWins,
    RightWins,
    Tie,
}

pub trait SelectionOperator {
    type FDim: D12;

    fn pressure(&self) -> usize {
        2
    }

    fn n_parents_per_crossover(&self) -> usize {
        2
    }

    fn feasibility_dominates<'a, ConstrDim>(
        p1: &Individual<'a, <Self::FDim as Dimension>::Smaller, ConstrDim>,
        p2: &Individual<'a, <Self::FDim as Dimension>::Smaller, ConstrDim>,
    ) -> DuelResult
    where
        <Self::FDim as Dimension>::Smaller: D01,
        ConstrDim: D01,
    {
        match (p1.is_feasible(), p2.is_feasible()) {
            (true, false) => DuelResult::LeftWins,
            (false, true) => DuelResult::RightWins,
            (false, false) => {
                let sum1 = p1.constraint_violation_totals.unwrap();
                let sum2 = p2.constraint_violation_totals.unwrap();
                if sum1 < sum2 {
                    DuelResult::LeftWins
                } else if sum1 > sum2 {
                    DuelResult::RightWins
                } else {
                    DuelResult::Tie
                }
            }
            (true, true) => DuelResult::Tie,
        }
    }
    /// Selects random participants from the population for the tournaments.
    /// If `n_crossovers * pressure` is greater than the population size, it will create multiple permutations
    /// to ensure there are enough random indices.
    fn select_participants(
        &self,
        population_size: usize,
        n_crossovers: usize,
        rng: &mut impl RandomGenerator,
    ) -> Vec<Vec<usize>> {
        // Note that we have fixed n_parents = 2 and pressure = 2
        let total_needed = n_crossovers * self.n_parents_per_crossover() * self.pressure();
        let mut all_indices = Vec::with_capacity(total_needed);

        let n_perms = total_needed.div_ceil(population_size); // Ceil division
        for _ in 0..n_perms {
            let mut perm: Vec<usize> = (0..population_size).collect();
            rng.shuffle_vec_usize(&mut perm);
            all_indices.extend_from_slice(&perm);
        }

        all_indices.truncate(total_needed);

        // Now split all_indices into chunks of size 2
        let mut result = Vec::with_capacity(n_crossovers);
        for chunk in all_indices.chunks(2) {
            // chunk is a slice of length 2
            result.push(vec![chunk[0], chunk[1]]);
        }

        result
    }

    /// Tournament between 2 individuals.
    fn tournament_duel<'a, ConstrDim>(
        &self,
        p1: &Individual<'a, <Self::FDim as Dimension>::Smaller, ConstrDim>,
        p2: &Individual<'a, <Self::FDim as Dimension>::Smaller, ConstrDim>,
        rng: &mut impl RandomGenerator,
    ) -> DuelResult
    where
        <Self::FDim as Dimension>::Smaller: D01,
        ConstrDim: D01;

    fn operate<ConstrDim>(
        &self,
        population: &Population<Self::FDim, ConstrDim>,
        n_crossovers: usize,
        rng: &mut impl RandomGenerator,
    ) -> (
        Population<Self::FDim, ConstrDim>,
        Population<Self::FDim, ConstrDim>,
    )
    where
        ConstrDim: D12,
        <ConstrDim as Dimension>::Smaller: D01,
        <Self::FDim as Dimension>::Smaller: D01,
    {
        let population_size = population.len();
        let participants = self.select_participants(population_size, n_crossovers, rng);
        let mut winners = Vec::with_capacity(n_crossovers);

        // For binary tournaments:
        // Each row of 'participants' is [p1, p2]
        for row in &participants {
            let ind_a = population.get(row[0]);
            let ind_b = population.get(row[1]);
            let duel_result = self.tournament_duel(&ind_a, &ind_b, rng);
            let winner = match duel_result {
                DuelResult::LeftWins => row[0],
                DuelResult::RightWins => row[1],
                DuelResult::Tie => row[1], // TODO: use random?
            };
            winners.push(winner);
        }

        // Split winners into two halves
        let mid = winners.len() / 2;
        let first_half = &winners[..mid];
        let second_half = &winners[mid..];

        // Create two new populations based on the split
        let population_a = population.selected(first_half);
        let population_b = population.selected(second_half);

        (population_a, population_b)
    }
}
