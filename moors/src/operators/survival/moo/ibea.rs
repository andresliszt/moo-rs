use ndarray::{Array1, Array2, ArrayView1, Axis};

use crate::genetic::{D12, PopulationMOO};
use crate::operators::SurvivalOperator;
use crate::random::RandomGenerator;

/// Indicator interface used by IBEA.
///
/// Math:
/// - IBEA fitness for population P:
///   F(x) = Σ_{y∈P, y≠x} -exp( -I(y,x)/κ )
/// - Pairwise matrix M[i,j] = -exp( -I(i,j)/κ ), with M[i,i]=0
/// - Then F[j] = Σ_i M[i,j]  (column-wise sum)
pub trait Indicator {
    /// Selection pressure κ (>0).
    fn kappa(&self) -> f64;

    /// Raw quality indicator I(f1,f2) (no exponent).
    fn indicator(&self, f1: ArrayView1<'_, f64>, f2: ArrayView1<'_, f64>) -> f64;

    /// Build M with M[i,j] = -exp( -I(i,j)/κ ), diag=0.
    /// Time: O(n^2·m).  Memory: O(n^2).
    fn pairwise_indicator(&self, fitness: &Array2<f64>) -> Array2<f64> {
        let n = fitness.nrows();
        let mut out = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            let ai = fitness.row(i);
            for j in 0..n {
                out[[i, j]] = if i == j {
                    0.0
                } else {
                    let bj = fitness.row(j);
                    -(-self.indicator(ai, bj) / self.kappa()).exp()
                };
            }
        }
        out
    }
}

/// Hypervolume indicator with fixed reference r (minimization).
///
/// HV formulas:
/// - HV({x}) = ∏_d (r_d - x_d)
/// - HV({a,b}) = HV(a)+HV(b) - ∏_d (r_d - max(a_d,b_d))
/// - I_HV(a,b) = HV({a,b}) - HV({b})
#[derive(Debug)]
pub struct HyperVolumeIndicator {
    reference: Array1<f64>,
    kappa: f64,
}

impl HyperVolumeIndicator {
    /// HV({point}) = ∏_d (r_d - point_d)
    fn hv_single(&self, point: ArrayView1<'_, f64>) -> f64 {
        self.reference
            .iter()
            .zip(point.iter())
            .map(|(r, x)| r - x)
            .product()
    }

    /// HV({f1,f2}) = HV(f1)+HV(f2) - ∏_d (r_d - max(f1_d, f2_d))
    fn hv_two(&self, f1: ArrayView1<'_, f64>, f2: ArrayView1<'_, f64>) -> f64 {
        let hv_f1 = self.hv_single(f1);
        let hv_f2 = self.hv_single(f2);
        let inter: f64 = self
            .reference
            .iter()
            .zip(f1.iter().zip(f2.iter()))
            .map(|(r, (a, b))| r - a.max(*b))
            .product();
        hv_f1 + hv_f2 - inter
    }
}

impl Indicator for HyperVolumeIndicator {
    fn kappa(&self) -> f64 {
        self.kappa
    }

    /// I_HV(f1,f2) = HV({f1,f2}) - HV({f2})
    fn indicator(&self, f1: ArrayView1<'_, f64>, f2: ArrayView1<'_, f64>) -> f64 {
        self.hv_two(f1, f2) - self.hv_single(f2)
    }
}

/// IBEA survival (environmental selection) driven by `Indicator`.
///
/// Loop:
/// 1) Build M[i,j] = -exp( -I(i,j)/κ ), diag=0
/// 2) F = Σ_i M[i,·]
/// 3) Repeat until |P| = α:
///    - k = argmin(F)
///    - remove k
///    - F ← F - M[k,·]    // removes k's contribution
///    - F[k] = +∞         // don't pick k again
#[derive(Debug, Clone)]
pub struct IbeaSurvivalOperator<I: Indicator> {
    indicator: I,
}

impl<I: Indicator> SurvivalOperator for IbeaSurvivalOperator<I> {
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
        let mut to_drop = population.len() - num_survive;
        let mut indices_to_drop: Vec<usize> = Vec::with_capacity(to_drop);

        // 1) Pairwise matrix M and 2) initial fitness F
        let m = self.indicator.pairwise_indicator(&population.fitness);
        // Compute F = Σ_i M[i,·] (column-wise sum of M).
        let mut f = m.sum_axis(Axis(0));

        // 3) Elimination loop
        while to_drop > 0 {
            // k = argmin(F)
            let k = f
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()) // assuming no NaNs
                .unwrap()
                .0;

            indices_to_drop.push(k);

            // F ← F - M[k,·]; avoid reselecting k
            f -= &m.row(k);
            f[k] = f64::INFINITY;

            to_drop -= 1;
        }
        // Define the indices that need to be selected
        let n = population.len();
        let mut dropped = vec![false; n];
        for i in indices_to_drop {
            dropped[i] = true;
        }
        let keep: Vec<usize> = (0..n).filter(|&i| !dropped[i]).collect();
        // Get the survival scorer, just the fitness IBEA
        let survival_score = f.select(Axis(0), &keep);
        let mut survivors = population.selected(&keep);
        survivors.set_survival_score(survival_score);
        survivors
    }
}

pub type IbeaHyperVolumeSurvivalOperator = IbeaSurvivalOperator<HyperVolumeIndicator>;

impl IbeaHyperVolumeSurvivalOperator {
    pub fn new(reference: Array1<f64>, kappa: f64) -> Self {
        Self {
            indicator: HyperVolumeIndicator {
                reference: reference,
                kappa: kappa,
            },
        }
    }
}
