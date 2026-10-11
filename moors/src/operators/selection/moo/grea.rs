//! GrEA mating (tournament) selection.
//!
//! Implements the tournament comparison rule described in Algorithm 2 of
//! **S. Yang, M. Li, X. Liu & J. Zheng, "A Grid-Based Evolutionary Algorithm
//! for Many-Objective Optimization", IEEE TEVC, 17(5):721-736, 2013.**
//!
//! Before dueling, `operate` recomputes a fresh, static grid-fitness
//! assignment over the *whole* current population (Algorithm 1's
//! per-generation "Fitness_assignment(P)" step) via [`fitness_assignment`].
//! Unlike
//! [`GreaSurvival`](crate::operators::survival::moo::GreaSurvival)'s
//! `grid_select`, this is a one-shot pairwise computation with no greedy
//! archive-building loop: Algorithm 2 only ever needs grid coordinates and
//! `GCD`. Each duel is then resolved, in order, by:
//! 1. Feasibility (constraint violation),
//! 2. Raw Pareto dominance,
//! 3. Grid dominance,
//! 4. Grid crowding distance `GCD` (smaller is better),
//! 5. Uniform random tie-break.
//!
//! This needs the whole-population context (grid bounds, `GCD`) *before*
//! any single duel is resolved, so this type implements [`SelectionOperator`]
//! directly (overriding `operate`) instead of the simpler
//! [`DuelSelection`](crate::operators::selection::DuelSelection) trait.
use ndarray::{Array1, Array2, Ix2};

use crate::{
    genetic::{D01, D12, PopulationMOO},
    non_dominated_sorting::dominates,
    operators::{
        grea::{GridBounds, grid_difference},
        selection::{DuelResult, SelectionOperator},
    },
    random::RandomGenerator,
};

/// Computes the paper's "Fitness_assignment(P)" (Algorithm 1, line 4): grid
/// coordinates and a static grid crowding distance `GCD` (Eq. 10) over the
/// whole candidate set, with no greedy archive-building process. Returns
/// `(gcd, grid_coords)` — GrEA's mating selection (Algorithm 2) only ever
/// reads these two values. Unlike
/// [`grid_select`](crate::operators::survival::moo::grea::grid_select),
/// there is no pick order and no archive: every individual's `GCD` is
/// computed relative to every other individual in `fitness`.
pub(crate) fn fitness_assignment(
    fitness: &Array2<f64>,
    div: usize,
) -> (Array1<f64>, Array2<usize>) {
    let n = fitness.nrows();
    let m = fitness.ncols();
    let bounds = GridBounds::new(fitness, div);
    let grid_coords = bounds.coordinates(fitness);

    let mut gcd = Array1::<f64>::zeros(n);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let gd = grid_difference(grid_coords.row(i), grid_coords.row(j));
            if gd < m {
                gcd[i] += (m - gd) as f64;
            }
        }
    }

    (gcd, grid_coords)
}

/// GrEA's grid-based binary tournament selection.
#[derive(Debug, Clone)]
pub struct GreaTournamentSelection {
    div: usize,
}

impl GreaTournamentSelection {
    /// `div` is the number of divisions per objective used to build the
    /// hyper-grid (paper's `div` hyper-parameter). Must match the value
    /// used by [`GreaSurvival`](crate::operators::survival::moo::GreaSurvival).
    pub fn new(div: usize) -> Self {
        Self { div }
    }

    /// Algorithm 2's tournament comparison rule for candidates `i` and `j`,
    /// given the `(gcd, grid_coords)` snapshot already computed for the
    /// *whole* current population. Not expressible as
    /// [`DuelSelection::tournament_duel`](crate::operators::selection::DuelSelection::tournament_duel)
    /// because that method only receives the two `Individual`s (fitness +
    /// scalar `rank`/`survival_score`) with no population index and no
    /// per-objective grid-coordinate vector, both of which are required
    /// here (see [`operate`](SelectionOperator::operate) below).
    fn duel_with_grid_context<ConstrDim>(
        &self,
        i: usize,
        j: usize,
        population: &PopulationMOO<ConstrDim>,
        gcd: &Array1<f64>,
        grid_coords: &Array2<usize>,
        rng: &mut impl RandomGenerator,
    ) -> usize
    where
        ConstrDim: D12,
        <ConstrDim as ndarray::Dimension>::Smaller: D01,
    {
        let ind_a = population.get(i);
        let ind_b = population.get(j);

        match Self::feasibility_dominates(&ind_a, &ind_b) {
            DuelResult::LeftWins => i,
            DuelResult::RightWins => j,
            DuelResult::Tie => {
                let fi = population.fitness.row(i);
                let fj = population.fitness.row(j);
                if dominates(fi, fj) {
                    return i;
                } else if dominates(fj, fi) {
                    return j;
                }

                let gi = grid_coords.row(i);
                let gj = grid_coords.row(j);
                if dominates(gi, gj) {
                    i
                } else if dominates(gj, gi) {
                    j
                } else if gcd[i] < gcd[j] {
                    i
                } else if gcd[j] < gcd[i] {
                    j
                } else if rng.gen_bool(0.5) {
                    i
                } else {
                    j
                }
            }
        }
    }
}

impl SelectionOperator for GreaTournamentSelection {
    type FDim = Ix2;

    fn operate<ConstrDim>(
        &self,
        population: &PopulationMOO<ConstrDim>,
        n_crossovers: usize,
        rng: &mut impl RandomGenerator,
    ) -> (PopulationMOO<ConstrDim>, PopulationMOO<ConstrDim>)
    where
        ConstrDim: D12,
        <ConstrDim as ndarray::Dimension>::Smaller: D01,
    {
        let population_size = population.len();
        // Algorithm 1's per-generation "Fitness_assignment(P)": a static grid
        // fitness assignment over the whole current population.
        let (gcd, grid_coords) = fitness_assignment(&population.fitness, self.div);

        let participants = self.select_participants(population_size, n_crossovers, rng);
        let mut winners = Vec::with_capacity(n_crossovers);

        for row in &participants {
            let (i, j) = (row[0], row[1]);
            let winner = self.duel_with_grid_context(i, j, population, &gcd, &grid_coords, rng);
            winners.push(winner);
        }

        let mid = winners.len() / 2;
        let population_a = population.selected(&winners[..mid]);
        let population_b = population.selected(&winners[mid..]);
        (population_a, population_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::MOORandomGenerator;
    use ndarray::array;

    #[test]
    fn test_grea_tournament_selection_operate() {
        let genes = array![[0.0], [1.0], [2.0], [3.0]];
        let fitness = array![[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]];
        let constraints = ndarray::Array2::<f64>::zeros((4, 0));
        let population = PopulationMOO::new(genes, fitness, constraints);

        let selector = GreaTournamentSelection::new(4);
        let mut rng = MOORandomGenerator::new_from_seed(Some(42));
        let (a, b) = selector.operate(&population, 2, &mut rng);

        assert_eq!(a.len() + b.len(), 4);
    }
}
