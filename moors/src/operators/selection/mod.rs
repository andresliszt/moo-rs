use crate::{
    genetic::{Individual, Population},
    operators::GeneticOperator,
    random::RandomGenerator,
};

pub mod random_tournament;
pub mod rank_and_survival_scoring_tournament;

// Enum to represent the result of a tournament duel.
#[derive(Debug, PartialEq, Eq)]
pub enum DuelResult {
    LeftWins,
    RightWins,
    Tie,
}

pub trait SelectionOperator: GeneticOperator {
    fn pressure(&self) -> usize {
        2
    }

    fn n_parents_per_crossover(&self) -> usize {
        2
    }

    /// Selects random participants from the population for the tournaments.
    /// If `n_crossovers * pressure` is greater than the population size, it will create multiple permutations
    /// to ensure there are enough random indices.
    fn _select_participants(
        &self,
        population_size: usize,
        n_crossovers: usize,
        rng: &mut dyn RandomGenerator,
    ) -> Vec<Vec<usize>> {
        // Note that we have fixed n_parents = 2 and pressure = 2
        let total_needed = n_crossovers * self.n_parents_per_crossover() * self.pressure();
        let mut all_indices = Vec::with_capacity(total_needed);

        let n_perms = (total_needed + population_size - 1) / population_size; // Ceil division
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
    fn tournament_duel(
        &self,
        p1: &Individual,
        p2: &Individual,
        rng: &mut dyn RandomGenerator,
    ) -> DuelResult;

    fn operate(
        &self,
        population: &Population,
        n_crossovers: usize,
        rng: &mut dyn RandomGenerator,
    ) -> (Population, Population) {
        let population_size = population.len();

        let participants = self._select_participants(population_size, n_crossovers, rng);

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
