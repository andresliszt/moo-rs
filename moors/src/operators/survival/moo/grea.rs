//! GrEA environmental selection.
//!
//! Implements Algorithm 4 ("Environmental_selection") of
//! **S. Yang, M. Li, X. Liu & J. Zheng, "A Grid-Based Evolutionary Algorithm
//! for Many-Objective Optimization", IEEE TEVC, 17(5):721-736, 2013.**
//!
//! Full, non-critical Pareto fronts are kept unmodified (their `rank` stays
//! the front index, exactly like NSGA-II). The last ("critical") front is
//! filled greedily using the grid-based ranking (`GR`), grid crowding
//! distance (`GCD`) and grid coordinate point distance (`GCPD`) computed by
//! [`grid_select`] (Algorithms 3, 5, 6 and 7): `order` alone decides who
//! survives. `survival_score` is left unset (`None`), like
//! [`Nsga3ReferencePointsSurvival`](crate::operators::survival::moo::Nsga3ReferencePointsSurvival)
//! and [`ReveaReferencePointsSurvival`](crate::operators::survival::moo::ReveaReferencePointsSurvival)
//! — nothing (not even
//! [`GreaTournamentSelection`](crate::operators::selection::moo::GreaTournamentSelection),
//! which computes its own unrelated static GCD) ever reads it back.
use ndarray::{Array1, Array2, ArrayView1};

use crate::{
    genetic::{D12, FrontsExt, PopulationMOO},
    non_dominated_sorting::{build_fronts, dominates},
    operators::{
        grea::{GridBounds, grid_difference},
        survival::SurvivalOperator,
    },
    random::RandomGenerator,
};

/// Algorithm 7's `Findout_best`: lexicographic argmin over `(GR, GCD, GCPD)`
/// among the not-yet-picked candidates.
fn findout_best(
    picked: &[bool],
    gr: &Array1<usize>,
    gcd: &Array1<f64>,
    gcpd: &Array1<f64>,
) -> usize {
    (0..picked.len())
        .filter(|&i| !picked[i])
        .min_by(|&a, &b| {
            gr[a]
                .cmp(&gr[b])
                .then_with(|| gcd[a].partial_cmp(&gcd[b]).unwrap())
                .then_with(|| gcpd[a].partial_cmp(&gcpd[b]).unwrap())
        })
        .expect("grid_select: no unpicked candidate left")
}

/// Algorithm 6's `GCD_calculation`: punishes every remaining candidate's
/// crowding distance based on its grid proximity to the just-picked `q`
/// (Eq. 10). `m` is the number of objectives, *not* `div`: the paper's
/// neighbourhood threshold and punishment are both expressed in terms of
/// the objective count `M`.
fn gcd_calculation(
    remaining: &[usize],
    q_coords: ArrayView1<usize>,
    grid_coords: &Array2<usize>,
    m: usize,
    gcd: &mut Array1<f64>,
) {
    for &p in remaining {
        let gd = grid_difference(grid_coords.row(p), q_coords);
        if gd < m {
            gcd[p] += (m - gd) as f64;
        }
    }
}

/// Algorithm 3's `GR_adjustment`: punishes every remaining candidate's grid
/// ranking based on its grid-dominance relation to the just-picked `q`.
/// `pd`/`excluded` are caller-owned scratch buffers (size `n`), reused across
/// iterations to avoid reallocating on every pick. `m` is the number of
/// objectives, *not* `div`: the paper expresses every punishment degree
/// (`M+2`, `M`, `M - GD`) in terms of the objective count `M`.
fn gr_adjustment(
    remaining: &[usize],
    q_coords: ArrayView1<usize>,
    grid_coords: &Array2<usize>,
    m: usize,
    gr: &mut Array1<usize>,
    pd: &mut [usize],
    excluded: &mut [bool],
) {
    let mut e_q: Vec<usize> = Vec::new();
    let mut g_q: Vec<usize> = Vec::new();
    let mut ng_q_minus_e: Vec<usize> = Vec::new();
    for &p in remaining {
        let gd = grid_difference(grid_coords.row(p), q_coords);
        if gd == 0 {
            e_q.push(p);
        } else if dominates(q_coords, grid_coords.row(p)) {
            g_q.push(p);
        } else {
            ng_q_minus_e.push(p);
        }
    }

    for &p in &e_q {
        gr[p] += m + 2;
    }
    for &p in &g_q {
        gr[p] += m;
    }

    for &p in remaining {
        pd[p] = 0;
    }
    for &p in e_q.iter().chain(g_q.iter()) {
        excluded[p] = true;
    }

    for &p in &ng_q_minus_e {
        let gd_pq = grid_difference(grid_coords.row(p), q_coords);
        if gd_pq < m {
            let cand = m - gd_pq;
            if pd[p] < cand {
                pd[p] = cand;
                let p_coords = grid_coords.row(p).to_owned();
                for &r in remaining {
                    if r == p || excluded[r] {
                        continue;
                    }
                    if dominates(p_coords.view(), grid_coords.row(r)) && pd[r] < pd[p] {
                        pd[r] = pd[p];
                    }
                }
            }
        }
    }

    for &p in &ng_q_minus_e {
        gr[p] += pd[p];
    }
    for &p in e_q.iter().chain(g_q.iter()) {
        excluded[p] = false;
    }
}

/// Runs GrEA's greedy grid-based selection (Algorithms 3, 5, 6 and 7 of the
/// paper) over `fitness`, picking `num_to_select` individuals one at a time
/// (best-first, by `(GR, GCD, GCPD)` lexicographic order) while dynamically
/// punishing grid-neighbours of the individual just picked. Returns the
/// picked indices (into `fitness`), best first; length is
/// `min(num_to_select, n)`.
///
/// This is exclusive to GrEA's environmental selection (survival): the
/// paper initializes `GCD` to zero (Algorithm 5) and only ever grows it
/// relative to individuals already picked into the archive (Algorithm 6).
/// Mating selection uses
/// [`fitness_assignment`](crate::operators::selection::moo::grea::fitness_assignment)
/// instead.
fn grid_select(fitness: &Array2<f64>, div: usize, num_to_select: usize) -> Vec<usize> {
    let n = fitness.nrows();
    let m = fitness.ncols();
    let bounds = GridBounds::new(fitness, div);
    let grid_coords = bounds.coordinates(fitness);

    // Initial GR(x) = sum_k Gk(x) (Eq. 9).
    let mut gr: Array1<usize> = grid_coords
        .rows()
        .into_iter()
        .map(|row| row.sum())
        .collect::<Vec<usize>>()
        .into();
    let mut gcd: Array1<f64> = Array1::zeros(n);
    let gcpd: Array1<f64> = (0..n)
        .map(|i| bounds.gcpd_of(fitness.row(i), grid_coords.row(i)))
        .collect::<Vec<f64>>()
        .into();

    let mut picked = vec![false; n];
    let mut pd = vec![0usize; n];
    let mut excluded = vec![false; n];
    let target = num_to_select.min(n);
    let mut order = Vec::with_capacity(target);

    for _ in 0..target {
        let q = findout_best(&picked, &gr, &gcd, &gcpd);
        picked[q] = true;
        order.push(q);

        let remaining: Vec<usize> = (0..n).filter(|&i| !picked[i]).collect();
        if remaining.is_empty() {
            break;
        }

        let q_coords = grid_coords.row(q);
        gcd_calculation(&remaining, q_coords, &grid_coords, m, &mut gcd);
        gr_adjustment(
            &remaining,
            q_coords,
            &grid_coords,
            m,
            &mut gr,
            &mut pd,
            &mut excluded,
        );
    }

    order
}

/// GrEA's grid-based environmental selection.
#[derive(Debug, Clone)]
pub struct GreaSurvival {
    div: usize,
}

impl GreaSurvival {
    /// `div` is the number of divisions per objective used to build the
    /// hyper-grid (paper's `div` hyper-parameter).
    pub fn new(div: usize) -> Self {
        Self { div }
    }
}

impl SurvivalOperator for GreaSurvival {
    type FDim = ndarray::Ix2;

    fn operate<ConstrDim>(
        &mut self,
        population: PopulationMOO<ConstrDim>,
        num_survive: usize,
        _rng: &mut impl RandomGenerator,
    ) -> PopulationMOO<ConstrDim>
    where
        ConstrDim: D12,
    {
        let mut fronts = build_fronts(population, num_survive);
        let critical = fronts
            .pop()
            .expect("build_fronts must return at least one front");

        let already_selected: usize = fronts.iter().map(|f| f.len()).sum();
        let remaining = num_survive - already_selected;

        // Algorithms 3, 5, 6 and 7: greedy grid-based selection of the
        // critical front. `order` alone decides who survives.
        let order = grid_select(&critical.fitness, self.div, remaining);
        let selected_critical = critical.selected(&order);

        fronts.push(selected_critical);
        fronts.to_population()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::NoopRandomGenerator;
    use ndarray::array;

    #[test]
    fn test_grid_select_selects_all_when_target_ge_n() {
        let fitness = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let order = grid_select(&fitness, 4, 4);
        assert_eq!(order.len(), 4);
        let mut sorted_order = order.clone();
        sorted_order.sort();
        assert_eq!(sorted_order, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_grid_select_partial_prefers_isolated_points() {
        // A valid non-dominated front: {0,1} are a near-duplicate pair at one
        // extreme, {2} is the other extreme, isolated from both.
        let fitness = array![[0.0, 10.0], [0.01, 9.99], [10.0, 0.0]];
        let order = grid_select(&fitness, 10, 2);
        assert_eq!(order.len(), 2);
        assert!(order.contains(&2));
    }

    #[test]
    fn test_grea_survival_keeps_num_survive_individuals() {
        let genes = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
        let fitness = array![[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0],];
        let constraints = ndarray::Array2::<f64>::zeros((5, 0));
        let population = PopulationMOO::new(genes, fitness, constraints);

        let mut survivor = GreaSurvival::new(4);
        let mut rng = NoopRandomGenerator::new();
        let survivors = survivor.operate(population, 3, &mut rng);

        assert_eq!(survivors.len(), 3);
        assert!(survivors.survival_score.is_none());
    }
}
