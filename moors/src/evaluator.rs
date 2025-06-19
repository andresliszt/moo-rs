//! # `evaluator` – From Genes to Population
//!
//! The **`Evaluator`** is the point where user‑supplied domain logic (fitness
//! and constraints functions) meets the core data structures of *moors*.  It
//! takes a 2‑D array of genomes (`PopulationGenes` = `Array2<f64>`) and returns
//! a fully populated [`Population`] with fitness values and optional constraints
use ndarray::{Array1, Array2, ArrayBase, Axis, Dimension, OwnedRepr};
use thiserror::Error;

use crate::genetic::{D01, D12, Population};

pub trait ConstraintsFn
where
    Self::Dim: D12,
    <Self::Dim as Dimension>::Smaller: D01,
{
    type Dim: D12;
    fn call(&self, genes: &Array2<f64>) -> ArrayBase<OwnedRepr<f64>, Self::Dim>;
}

impl<G, Dim> ConstraintsFn for G
where
    G: Fn(&Array2<f64>) -> ArrayBase<OwnedRepr<f64>, Dim>,
    Dim: D12,
    <Dim as Dimension>::Smaller: D01,
{
    type Dim = Dim;
    fn call(&self, genes: &Array2<f64>) -> ArrayBase<OwnedRepr<f64>, Dim> {
        (self)(genes)
    }
}

/// A zero-sized type that serves as the default implementation of ConstraintsFn.
#[derive(Debug)]
pub struct NoConstraints;

/// Implement the ConstraintsFn trait for the default case:
/// - Associated type `Dim` is fixed to `Ix1`
impl ConstraintsFn for NoConstraints {
    type Dim = ndarray::Ix1;

    fn call(&self, _genes: &Array2<f64>) -> ArrayBase<OwnedRepr<f64>, Self::Dim> {
        Array1::from(vec![])
    }
}

pub trait FitnessFn
where
    <Self::Dim as Dimension>::Smaller: D01,
{
    type Dim: D12;
    fn call(&self, genes: &Array2<f64>) -> ArrayBase<OwnedRepr<f64>, Self::Dim>;
}

impl<F, Dim> FitnessFn for F
where
    F: Fn(&Array2<f64>) -> ArrayBase<OwnedRepr<f64>, Dim>,
    Dim: D12,
    <Dim as Dimension>::Smaller: D01,
{
    type Dim = Dim;
    fn call(&self, genes: &Array2<f64>) -> ArrayBase<OwnedRepr<f64>, Dim> {
        (self)(genes)
    }
}

/// Error type for the Evaluator.
#[derive(Debug, Error)]
pub enum EvaluatorError {
    #[error("No feasible individuals found in the population.")]
    NoFeasibleIndividuals,
}

/// Evaluator struct for calculating fitness and (optionally) constraints,
/// then assembling a `Population`. In addition to the user-provided constraints function,
/// optional lower and upper bounds can be specified for the decision variables (genes).
#[derive(Debug)]
pub struct Evaluator<F, G>
where
    F: FitnessFn,
    G: ConstraintsFn,
{
    fitness: F,
    constraints: G,
    keep_infeasible: bool,
    /// Optional lower bound for each gene.
    lower_bound: Option<f64>,
    /// Optional upper bound for each gene.
    upper_bound: Option<f64>,
}

impl<F, G> Evaluator<F, G>
where
    F: FitnessFn,
    G: ConstraintsFn,
{
    /// Creates a new `Evaluator` with a fitness function, an optional constraints function,
    /// a flag to keep infeasible individuals, and optional lower/upper bounds.
    pub fn new(
        fitness: F,
        constraints: G,
        keep_infeasible: bool,
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
    ) -> Self {
        Self {
            fitness,
            constraints,
            keep_infeasible,
            lower_bound,
            upper_bound,
        }
    }

    /// Builds the population instance from the genes. If `keep_infeasible` is false,
    /// individuals are filtered out if they do not satisfy:
    ///   - The provided constraints function (all constraint values must be ≤ 0), and
    ///   - The optional lower and upper bounds (each gene must satisfy lower_bound <= gene <= upper_bound).
    pub fn evaluate(
        &self,
        genes: Array2<f64>,
    ) -> Result<Population<F::Dim, G::Dim>, EvaluatorError> {
        let fitness = self.fitness.call(&genes);
        let constraints = self.constraints.call(&genes);
        let mut evaluated_population = Population {
            genes: genes,
            fitness: fitness,
            constraints: constraints,
            rank: None,
            survival_score: None,
        };

        if !self.keep_infeasible {
            // Create a list of all indices.
            let n = evaluated_population.genes.nrows();
            let mut feasible_indices: Vec<usize> = (0..n).collect();

            // Filter individuals that do not satisfy the constraints function (if provided).
            if evaluated_population.constraints.len() > 0 {
                feasible_indices.retain(|&i| {
                    evaluated_population
                        .constraints
                        .index_axis(Axis(0), i)
                        .iter()
                        .all(|&val| val <= 0.0)
                });
            };

            // Further filter individuals based on the optional lower and upper bounds.
            if self.lower_bound.is_some() || self.upper_bound.is_some() {
                feasible_indices.retain(|&i| {
                    let individual = evaluated_population.genes.index_axis(Axis(0), i);
                    let lower_ok = self
                        .lower_bound
                        .map_or(true, |lb| individual.iter().all(|&x| x >= lb));
                    let upper_ok = self
                        .upper_bound
                        .map_or(true, |ub| individual.iter().all(|&x| x <= ub));

                    lower_ok && upper_ok
                });
            }

            if feasible_indices.is_empty() {
                return Err(EvaluatorError::NoFeasibleIndividuals);
            }

            // Filter all relevant arrays (genes, fitness, and constraints if present).
            evaluated_population = evaluated_population.selected(&feasible_indices);
        }

        Ok(evaluated_population)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, Axis, array, concatenate};

    // ──────────────────────────────────────────────────────────────────────────
    // Helper functions
    // ──────────────────────────────────────────────────────────────────────────

    /// One-objective fitness (sphere) → N × 1 column
    fn fitness_2d_single(genes: &Array2<f64>) -> Array2<f64> {
        genes // genes: N × d
            .map_axis(Axis(1), |ind| ind.iter().map(|&x| x * x).sum::<f64>())
            .insert_axis(Axis(1)) // N × 1
    }

    /// Same sphere but returned as a flat vector (N elements)
    fn fitness_1d(genes: &Array2<f64>) -> Array1<f64> {
        genes.map_axis(Axis(1), |ind| ind.iter().map(|&x| x * x).sum::<f64>())
    }

    /// Two objectives:
    ///   f₀ = Σ x²   |   f₁ = Σ |x|
    fn fitness_2d_two_obj(genes: &Array2<f64>) -> Array2<f64> {
        let f0 = genes
            .map_axis(Axis(1), |ind| ind.iter().map(|&x| x * x).sum::<f64>())
            .insert_axis(Axis(1));
        let f1 = genes
            .map_axis(Axis(1), |ind| ind.iter().map(|&x| x.abs()).sum::<f64>())
            .insert_axis(Axis(1));
        concatenate![Axis(1), f0, f1] // N × 2
    }

    /// Multiple constraints:
    ///   c₀ = Σx – 10  (≤ 0)              • global resource limit
    ///   cᵢ = –xᵢ        (≤ 0)            • every gene must be ≥ 0
    fn constraints_multi(genes: &Array2<f64>) -> Array2<f64> {
        let c0 = genes
            .sum_axis(Axis(1))
            .mapv(|s| s - 10.0)
            .insert_axis(Axis(1));
        let non_neg = genes.mapv(|x| -x);
        concatenate![Axis(1), c0, non_neg] // N × (d+1)
    }

    /// Single constraint: c = Σx – 10 ≤ 0
    fn constraints_single(genes: &Array2<f64>) -> Array1<f64> {
        genes.sum_axis(Axis(1)).mapv(|s| s - 10.0)
    }

    fn no_constraints(_genes: &Array2<f64>) -> Array1<f64> {
        Array1::from(vec![])
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 2-D fitness – no constraints
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn two_d_fitness_without_constraints_keeps_every_row() {
        let eval = Evaluator::new(
            fitness_2d_single,
            no_constraints,
            /* keep_infeasible = */ true,
            None,
            None,
        );

        let genes = array![[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]];
        let fit = eval.evaluate(genes).unwrap().fitness;

        // Each row keeps its sphere value in a single column
        let expected = array![[5.0], [25.0], [0.0]];
        assert_eq!(fit, expected);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 2-D fitness + constraints
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn multi_constraints_are_computed_correctly() {
        let eval = Evaluator::new(fitness_2d_single, constraints_multi, true, None, None);

        let genes = array![
            /* idx 0 */ [1.0, 2.0], // Σ=3  → c0 = -7 ; all genes ≥0
            /* idx 1 */ [3.0, 4.0], // Σ=7  → c0 = -3
            /* idx 2 */
            [5.0, 6.0], // Σ=11 → c0 =  1 (will be infeasible when filtering)
        ];
        let c = eval.evaluate(genes).unwrap().constraints;

        let expected = array![[-7.0, -1.0, -2.0], [-3.0, -3.0, -4.0], [1.0, -5.0, -6.0],];
        assert_eq!(c, expected);
    }

    #[test]
    fn keep_infeasible_true_retains_every_row() {
        let eval = Evaluator::new(
            fitness_2d_single,
            constraints_multi,
            /* keep_infeasible = */ true, // do NOT filter
            None,
            None,
        );

        let genes = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let pop = eval.evaluate(genes).unwrap();
        assert_eq!(
            pop.genes.nrows(),
            3,
            "Nothing filtered when keep_infeasible = true"
        );
    }

    #[test]
    fn infeasible_or_out_of_bounds_are_dropped() {
        // Rule set: constraints ≤0   AND   0 ≤ gene ≤ 5
        let eval = Evaluator::new(
            fitness_2d_single,
            constraints_multi,
            /* keep_infeasible */ false,
            Some(0.0), // lower
            Some(5.0), // upper
        );

        let genes = array![
            /* 0 OK  */ [1.0, 2.0], // inside bounds, constraints satisfied
            /* 1 OK  */ [3.0, 4.0], // inside bounds, constraints satisfied
            /* 2 BAD */ [6.0, 1.0], // 6.0 violates upper bound
        ];
        let pop = eval.evaluate(genes).unwrap();
        assert_eq!(
            pop.genes.nrows(),
            2,
            "Row with 6.0 was removed for exceeding upper_bound"
        );
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 1-D fitness + multi-column constraints
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn one_d_fitness_multi_constraints_filters_by_constraint_only() {
        let eval = Evaluator::new(
            fitness_1d,        // Array2 → Array1
            constraints_multi, // Array2 → Array2
            false,
            None,
            None,
        );

        let genes = array![
            /* 0 OK  */ [1.0, 2.0], // Σ=3  (feasible)
            /* 1 BAD */ [6.0, 5.0], // Σ=11 (violates c0)
        ];
        let pop = eval.evaluate(genes).unwrap();

        assert_eq!(pop.genes.nrows(), 1, "Second row violates Σx - 10 ≤ 0");
        assert_eq!(pop.fitness, array![5.0]);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 1-D fitness + single constraint
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn one_d_fitness_single_constraint_filters_correctly() {
        let eval = Evaluator::new(
            fitness_1d,
            constraints_single, // Array2 → Array1
            false,
            None,
            None,
        );

        let genes = array![
            /* 0 OK  */ [2.0, 3.0], // Σ=5  (feasible)
            /* 1 BAD */ [5.0, 6.0], // Σ=11 (violates)
        ];
        let pop = eval.evaluate(genes).unwrap();

        assert_eq!(pop.genes.nrows(), 1);
        assert_eq!(pop.fitness, array![13.0]); // 2²+3²=13
        assert_eq!(
            pop.constraints,
            array![-5.0], // Σ - 10 = -5
            "Only constraint column for the surviving row"
        );
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 1-D fitness – bounds filtering without constraints
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn bounds_only_filtering_removes_rows_outside_range() {
        let eval = Evaluator::new(fitness_1d, no_constraints, false, Some(0.0), Some(5.0));

        let genes = array![
            /* 0 OK  */ [1.0, 2.0],
            /* 1 OK  */ [3.0, 4.0],
            /* 2 BAD */ [6.0, 0.5], // 6.0 exceeds upper bound
        ];
        let pop = eval.evaluate(genes).unwrap();
        assert_eq!(
            pop.genes.nrows(),
            2,
            "Third row removed solely due to upper_bound violation"
        );
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 1-D fitness – everything filtered → EvaluatorError
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn all_rows_removed_returns_no_feasible_error() {
        let eval = Evaluator::new(fitness_1d, constraints_multi, false, Some(0.0), Some(5.0));

        // Every candidate violates either constraint (Σ>10) or bounds (6.0)
        let genes = array![[6.0, 6.0], [5.5, 6.0], [6.0, 4.0]];
        let err = eval.evaluate(genes).unwrap_err();
        assert!(
            matches!(err, EvaluatorError::NoFeasibleIndividuals),
            "When no rows survive, Evaluator must return the dedicated error"
        );
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 2-objective fitness – no constraints
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn two_objective_fitness_is_computed_for_each_row() {
        let eval = Evaluator::new(fitness_2d_two_obj, no_constraints, true, None, None);

        let genes = array![[1.0, 2.0], [3.0, 4.0]];
        let fit = eval.evaluate(genes).unwrap().fitness;

        // Row-wise expected values: [Σx², Σ|x|]
        let expected = array![[5.0, 3.0], [25.0, 7.0]];
        assert_eq!(fit, expected);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 2-objective fitness + constraints + bounds
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn two_objective_with_filtering_keeps_only_feasible_and_in_bounds() {
        let eval = Evaluator::new(
            fitness_2d_two_obj,
            constraints_multi,
            false,
            Some(0.0),
            Some(5.0),
        );

        let genes = array![
            /* 0 OK  */ [1.0, 2.0], // constraints OK, bounds OK
            /* 1 OK  */ [3.0, 4.0], // idem
            /* 2 BAD */ [6.0, 1.0], // 6.0 violates upper bound
            /* 3 BAD */ [4.0, 7.0], // Σ>10 violates c0
        ];
        let pop = eval.evaluate(genes).unwrap();

        assert_eq!(
            pop.genes.nrows(),
            2,
            "Rows 2 and 3 filtered for bound and constraint violations"
        );

        let expected_fit = array![[5.0, 3.0], [25.0, 7.0]];
        assert_eq!(pop.fitness, expected_fit);
    }
}
