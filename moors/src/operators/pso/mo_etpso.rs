//! Concrete velocity-update and merit operators used by
//! [`MoEtpso`](crate::algorithms::MoEtpso) (Fitas, 2024).
use ndarray::{Array1, Array2, Axis};

use super::merit::MeritOperator;
use super::velocity::VelocityUpdateOperator;
use crate::genetic::{D12, PopulationMOO};
use crate::non_dominated_sorting::fast_non_dominated_sorting;
use crate::random::RandomGenerator;

/// Classical constriction-factor PSO velocity update (Clerc & Kennedy, 2002),
/// as used by MO-ETPSO (Fitas, 2024, eq. 5-6):
///
/// `v_new = K * [ w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x) ]`
///
/// with `K = 2 / |2 - phi - sqrt(phi^2 - 4*phi)|` and `phi = c1 + c2`. `phi`
/// must be `> 4` for `K` to guarantee convergence; otherwise this falls back
/// to `K = 1` (plain inertia-weighted PSO).
#[derive(Debug, Clone, Copy)]
pub struct ConstrictionVelocityUpdate {
    pub inertia_weight: f64,
    pub cognitive_coefficient: f64,
    pub social_coefficient: f64,
}

impl ConstrictionVelocityUpdate {
    pub fn new(inertia_weight: f64, cognitive_coefficient: f64, social_coefficient: f64) -> Self {
        Self {
            inertia_weight,
            cognitive_coefficient,
            social_coefficient,
        }
    }

    fn constriction_factor(&self) -> f64 {
        let phi = self.cognitive_coefficient + self.social_coefficient;
        if phi <= 4.0 {
            return 1.0;
        }
        2.0 / (2.0 - phi - (phi * phi - 4.0 * phi).sqrt()).abs()
    }
}

impl VelocityUpdateOperator for ConstrictionVelocityUpdate {
    fn operate(
        &self,
        positions: &Array2<f64>,
        velocities: &Array2<f64>,
        pbest_positions: &Array2<f64>,
        gbest_position: &Array2<f64>,
        rng: &mut impl RandomGenerator,
    ) -> Array2<f64> {
        let k = self.constriction_factor();
        let n = positions.nrows();
        let d = positions.ncols();
        let mut new_velocities = Array2::<f64>::zeros((n, d));

        for i in 0..n {
            let r1 = rng.gen_proability();
            let r2 = rng.gen_proability();
            for j in 0..d {
                let cognitive =
                    self.cognitive_coefficient * r1 * (pbest_positions[[i, j]] - positions[[i, j]]);
                let social =
                    self.social_coefficient * r2 * (gbest_position[[0, j]] - positions[[i, j]]);
                new_velocities[[i, j]] =
                    k * (self.inertia_weight * velocities[[i, j]] + cognitive + social);
            }
        }
        new_velocities
    }
}

/// Default merit used by MO-ETPSO (Fitas, 2024): `max(rank) - rank +
/// crowding - constraint_violation`. Non-dominated (rank 0) solutions always
/// beat dominated ones, ties are broken by crowding (diversity), and any
/// constraint violation only ever pulls the score down.
#[derive(Debug, Clone, Copy, Default)]
pub struct RankCrowdingMerit;

impl RankCrowdingMerit {
    pub fn new() -> Self {
        Self
    }
}

impl MeritOperator for RankCrowdingMerit {
    fn compute<ConstrDim>(&self, population: &PopulationMOO<ConstrDim>) -> Array1<f64>
    where
        ConstrDim: D12,
    {
        let n = population.len();
        let fronts_idx = fast_non_dominated_sorting(&population.fitness, n);

        let mut rank = Array1::<f64>::zeros(n);
        let mut crowding = Array1::<f64>::zeros(n);

        for (front_rank, indices) in fronts_idx.iter().enumerate() {
            for &i in indices {
                rank[i] = front_rank as f64;
            }
            let front_fitness = population.fitness.select(Axis(0), indices);
            let front_crowding = product_crowding_distance(&front_fitness);
            for (k, &i) in indices.iter().enumerate() {
                crowding[i] = front_crowding[k];
            }
        }

        let max_rank = rank.iter().cloned().fold(0.0_f64, f64::max);
        let n = population.len();
        let violation = population
            .constraint_violation_totals
            .clone()
            .unwrap_or_else(|| Array1::zeros(n));

        Array1::from_shape_fn(n, |i| (max_rank - rank[i]) + crowding[i] - violation[i])
    }
}

/// Product-based crowding distance, as defined by MO-ETPSO (eq. 3): for each
/// objective, multiplies (instead of summing, like classic NSGA-II) the
/// normalized gap between an individual's neighbours. Boundary individuals of
/// a front always get [`MAX_CROWDING`].
///
/// Values are intentionally bounded to `[0, MAX_CROWDING]` (rather than
/// literal infinity for boundary points): the merit score adds crowding on
/// top of a rank term, and an unbounded crowding value would let diversity
/// override Pareto rank, which must never happen — a dominated individual
/// must never outscore a non-dominated one.
const MAX_CROWDING: f64 = 1.0;

fn product_crowding_distance(front_fitness: &Array2<f64>) -> Array1<f64> {
    let n = front_fitness.nrows();
    let m = front_fitness.ncols();

    if n <= 2 {
        return Array1::from_elem(n, MAX_CROWDING);
    }

    let mut distances = Array1::<f64>::ones(n);
    let mut is_boundary = vec![false; n];

    for obj in 0..m {
        let col = front_fitness.column(obj);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| col[a].partial_cmp(&col[b]).unwrap());

        is_boundary[order[0]] = true;
        is_boundary[order[n - 1]] = true;

        let min_v = col[order[0]];
        let max_v = col[order[n - 1]];
        let range = max_v - min_v;

        if range > 0.0 {
            for k in 1..(n - 1) {
                let next = col[order[k + 1]];
                let prev = col[order[k - 1]];
                distances[order[k]] *= (next - prev) / range;
            }
        }
    }

    for i in 0..n {
        if is_boundary[i] {
            distances[i] = MAX_CROWDING;
        }
    }
    distances
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::MOORandomGenerator;
    use ndarray::array;

    #[test]
    fn constriction_factor_matches_clerc_kennedy_defaults() {
        // c1 = c2 = 2.05 -> phi = 4.1, K ~ 0.7298 (well-known constant).
        let op = ConstrictionVelocityUpdate::new(1.0, 2.05, 2.05);
        let k = op.constriction_factor();
        assert!((k - 0.729_843_788_128_36).abs() < 1e-9, "K = {}", k);
    }

    #[test]
    fn zero_distance_to_pbest_and_gbest_only_applies_inertia() {
        let op = ConstrictionVelocityUpdate::new(0.7, 2.05, 2.05);
        let positions = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let velocities = Array2::from_shape_vec((1, 2), vec![0.5, -0.5]).unwrap();
        let pbest = positions.clone();
        let gbest = positions.clone();

        let mut rng = MOORandomGenerator::new_from_seed(Some(1));
        let new_v = op.operate(&positions, &velocities, &pbest, &gbest, &mut rng);

        let k = op.constriction_factor();
        assert!((new_v[[0, 0]] - k * 0.7 * 0.5).abs() < 1e-12);
        assert!((new_v[[0, 1]] - k * 0.7 * -0.5).abs() < 1e-12);
    }

    #[test]
    fn rank_crowding_merit_prefers_non_dominated_and_diverse() {
        let genes = array![[0.0], [0.0], [0.0], [0.0]];
        let fitness = array![[1.0, 2.0], [2.0, 1.0], [1.5, 1.5], [3.0, 3.0]];
        let population: PopulationMOO = PopulationMOO::new_unconstrained(genes, fitness);

        let merit = RankCrowdingMerit::new().compute(&population);

        // Individual 3 ([3.0, 3.0]) is dominated by the other three, so it
        // must have the lowest merit.
        let min_idx = merit
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(min_idx, 3);
    }
}
