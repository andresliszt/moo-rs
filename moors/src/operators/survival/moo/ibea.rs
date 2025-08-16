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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genetic::PopulationMOO;
    use crate::random::NoopRandomGenerator;
    use ndarray::{Array1, Array2, array};

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    // ---------------------------
    // Indicator (math) unit tests
    // ---------------------------

    #[test]
    /// For r = (3,3), a = (1,1), b = (2,2):
    ///  HV(a)=4, HV(b)=1, HV({a,b})=4, so:
    ///  I(a,b)=HV({a,b})-HV(b)=3, I(b,a)=HV({a,b})-HV(a)=0.
    fn indicator_hv_basics() {
        let r: Array1<f64> = array![3.0, 3.0];
        let ind = HyperVolumeIndicator {
            reference: r,
            kappa: 1.0,
        };

        let a = array![1.0, 1.0];
        let b = array![2.0, 2.0];

        let i_ab = ind.indicator(a.view(), b.view());
        let i_ba = ind.indicator(b.view(), a.view());

        assert!(approx_eq(i_ab, 3.0, 1e-12));
        assert!(approx_eq(i_ba, 0.0, 1e-12));
    }

    #[test]
    /// M[i,j] = -exp( -I(i,j)/kappa ), diag = 0.
    /// With kappa=1 and the previous points:
    ///  I(a,b)=3  => M[a,b] = -exp(-3)
    ///  I(b,a)=0  => M[b,a] = -exp(0) = -1
    fn pairwise_indicator_matrix_values() {
        let r: Array1<f64> = array![3.0, 3.0];
        let ind = HyperVolumeIndicator {
            reference: r,
            kappa: 1.0,
        };

        let fitness: Array2<f64> = array![[1.0, 1.0], [2.0, 2.0]]; // rows: a, b
        let m = ind.pairwise_indicator(&fitness);

        assert!(approx_eq(m[[0, 0]], 0.0, 1e-12));
        assert!(approx_eq(m[[1, 1]], 0.0, 1e-12));
        assert!(approx_eq(m[[0, 1]], -(-3.0f64).exp(), 1e-12)); // -e^{-3}
        assert!(approx_eq(m[[1, 0]], -1.0, 1e-12));
        // Bounds sanity: off-diagonals in (-1, 0]
        for i in 0..2 {
            for j in 0..2 {
                if i != j {
                    assert!(m[[i, j]] <= 0.0 && m[[i, j]] > -1.0000001);
                }
            }
        }
    }

    // ---------------------------------------
    // Survival operator (IBEA) behavior tests
    // ---------------------------------------

    #[test]
    /// If num_survive == population size, the operator should return the same population
    /// (genes/fitness unchanged) and set a survival_score with matching length.
    fn operate_no_drop_returns_same_population() {
        let r: Array1<f64> = array![3.0, 3.0];
        let mut op = IbeaHyperVolumeSurvivalOperator::new(r, 1.0);

        let genes: Array2<f64> = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let fitness: Array2<f64> = array![[1.0, 2.0], [2.0, 1.0], [2.0, 2.0]];
        let pop = PopulationMOO::new_unconstrained(genes.clone(), fitness.clone());

        let mut rng = NoopRandomGenerator::new();
        let out = op.operate(pop, 3, &mut rng);

        assert_eq!(out.len(), 3);
        assert_eq!(out.genes, genes);
        assert_eq!(out.fitness, fitness);

        let score = out.survival_score.as_ref().expect("survival score set");
        assert_eq!(score.len(), 3);
        assert!(score.iter().all(|v| v.is_finite()));
    }

    #[test]
    /// Simple elimination scenario.
    ///
    /// r=(3,3), kappa=1, fitness rows:
    ///   a=(1,1), b=(2,2), c=(2.5,2.5)
    ///
    /// Pairwise I:
    ///   I(a,b)=3,    I(b,a)=0
    ///   I(a,c)=3.75, I(c,a)=0
    ///   I(b,c)=0.75, I(c,b)=0
    ///
    /// M[i,j] = -exp(-I(i,j)):
    ///   M[a,b]=-e^{-3},  M[b,a]=-1
    ///   M[a,c]=-e^{-3.75}, M[c,a]=-1
    ///   M[b,c]=-e^{-0.75}, M[c,b]=-1
    ///
    /// Column sums F[j] = Σ_i M[i,j]:
    ///   F[a] = -1 + -1 = -2
    ///   F[b] = -e^{-3} + -1 ≈ -1.049787
    ///   F[c] = -e^{-3.75} + -e^{-0.75} ≈ -0.495884
    ///
    /// The operator removes argmin(F). With num_survive=2 it drops index of 'a' first
    /// and keeps 'b' and 'c'. This asserts that behavior.
    fn operate_drops_one_keeps_two_expected_indices() {
        let r: Array1<f64> = array![3.0, 3.0];
        let mut op = IbeaHyperVolumeSurvivalOperator::new(r, 1.0);

        // genes are arbitrary but aligned with fitness rows (a, b, c)
        let genes: Array2<f64> = array![[10.0, 10.0], [20.0, 20.0], [25.0, 25.0]];
        let fitness: Array2<f64> = array![[1.0, 1.0], [2.0, 2.0], [2.5, 2.5]];

        let pop = PopulationMOO::new_unconstrained(genes.clone(), fitness.clone());
        let mut rng = NoopRandomGenerator::new();
        let out = op.operate(pop, 2, &mut rng);

        // Expect 2 survivors: rows corresponding to b and c (indices 1 and 2).
        assert_eq!(out.len(), 2);
        assert_eq!(out.genes, array![[20.0, 20.0], [25.0, 25.0]]);
        assert_eq!(out.fitness, array![[2.0, 2.0], [2.5, 2.5]]);

        // Survival score present and sized correctly
        let score = out.survival_score.as_ref().expect("survival score set");
        assert_eq!(score.len(), 2);
        assert!(score.iter().all(|v| v.is_finite()));
    }

    #[test]
    /// Smoke test on asymmetry and diagonal of the pairwise matrix for a larger set.
    fn pairwise_matrix_shape_and_diagonal() {
        let r: Array1<f64> = array![3.0, 3.0, 3.0];
        let ind = HyperVolumeIndicator {
            reference: r,
            kappa: 0.5,
        };

        let fitness: Array2<f64> = array![[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [2.5, 1.5, 2.0]];

        let m = ind.pairwise_indicator(&fitness);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 3);

        // Diagonal must be exactly zero; off-diagonal must be in (-1, 0]
        for i in 0..3 {
            assert!(approx_eq(m[[i, i]], 0.0, 1e-12));
            for j in 0..3 {
                if i != j {
                    assert!(m[[i, j]] <= 0.0 && m[[i, j]] > -1.0000001);
                }
            }
        }
    }
}
