//! # `evaluator` – From Genes to Population
//!
//! The **`Evaluator`** is the point where user‑supplied domain logic (fitness
//! and constraints functions) meets the core data structures of *moors*.  It
//! takes a 2‑D array of genomes (`PopulationGenes` = `Array2<f64>`) and returns
//! a fully populated [`Population`] with fitness values and optional constraints
use derive_builder::Builder;
use ndarray::{Array2, ArrayBase, Axis, Dimension, OwnedRepr};
use thiserror::Error;

use crate::genetic::{D01, D12, Population};

pub trait ConstraintsFn
where
    Self::Dim: D12,
    <Self::Dim as Dimension>::Smaller: D01,
{
    type Dim: D12;
    fn call(&self, genes: &Array2<f64>, context_id: usize) -> ArrayBase<OwnedRepr<f64>, Self::Dim>;
    fn lower_bound(&self) -> Option<f64> {
        None
    }
    fn upper_bound(&self) -> Option<f64> {
        None
    }
}

impl<G, Dim> ConstraintsFn for G
where
    G: Fn(&Array2<f64>, usize) -> ArrayBase<OwnedRepr<f64>, Dim>,
    Dim: D12,
    <Dim as Dimension>::Smaller: D01,
{
    type Dim = Dim;
    fn call(&self, genes: &Array2<f64>, context_id: usize) -> ArrayBase<OwnedRepr<f64>, Dim> {
        self(genes, context_id)
    }
}

/// A zero-sized type that serves as the default implementation of ConstraintsFn.
#[derive(Debug)]
pub struct NoConstraints;

/// Implement the ConstraintsFn trait for the default case:
/// - Associated type `Dim` is fixed to `Ix1`
impl ConstraintsFn for NoConstraints {
    type Dim = ndarray::Ix2;

    fn call(
        &self,
        genes: &Array2<f64>,
        _context_id: usize,
    ) -> ArrayBase<OwnedRepr<f64>, Self::Dim> {
        let n = genes.nrows();
        Array2::zeros((n, 0))
    }
}

pub trait FitnessFn
where
    <Self::Dim as Dimension>::Smaller: D01,
{
    type Dim: D12;
    fn call(&self, genes: &Array2<f64>, context_id: usize) -> ArrayBase<OwnedRepr<f64>, Self::Dim>;
}

impl<F, Dim> FitnessFn for F
where
    F: Fn(&Array2<f64>, usize) -> ArrayBase<OwnedRepr<f64>, Dim>,
    Dim: D12,
    <Dim as Dimension>::Smaller: D01,
{
    type Dim = Dim;
    fn call(&self, genes: &Array2<f64>, context_id: usize) -> ArrayBase<OwnedRepr<f64>, Dim> {
        self(genes, context_id)
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
#[derive(Debug, Builder)]
#[builder(pattern = "owned")]
pub struct Evaluator<F, G>
where
    F: FitnessFn,
    G: ConstraintsFn,
{
    fitness: F,
    constraints: G,
    #[builder(default = "true")]
    keep_infeasible: bool,
}

impl<F, G> Evaluator<F, G>
where
    F: FitnessFn,
    G: ConstraintsFn,
{
    /// Builds the population instance from the genes. If `keep_infeasible` is false,
    /// individuals are filtered out if they do not satisfy:
    ///   - The provided constraints function (all constraint values must be ≤ 0), and
    ///   - The optional lower and upper bounds (each gene must satisfy lower_bound <= gene <= upper_bound).
    pub fn evaluate(
        &self,
        genes: Array2<f64>,
        context_id: usize,
    ) -> Result<Population<F::Dim, G::Dim>, EvaluatorError> {
        let fitness = self.fitness.call(&genes, context_id);
        let constraints = self.constraints.call(&genes, context_id);
        let mut evaluated_population = Population::new(genes, fitness, constraints);

        if !self.keep_infeasible {
            // Create a list of all indices.
            let n = evaluated_population.genes.nrows();
            let mut feasible_indices: Vec<usize> = (0..n).collect();

            // Filter individuals that do not satisfy the constraints function (if provided).
            if !evaluated_population.constraints.is_empty() {
                feasible_indices.retain(|&i| {
                    evaluated_population
                        .constraints
                        .index_axis(Axis(0), i)
                        .iter()
                        .all(|&val| val <= 0.0)
                });
            };
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

    use crate::NoConstraints;

    // ──────────────────────────────────────────────────────────────────────────
    // Helper functions
    // ──────────────────────────────────────────────────────────────────────────

    /// One-objective fitness (sphere) → N × 1 column
    fn fitness_2d_single(genes: &Array2<f64>, _context_id: usize) -> Array2<f64> {
        genes // genes: N × d
            .map_axis(Axis(1), |ind| ind.iter().map(|&x| x * x).sum::<f64>())
            .insert_axis(Axis(1)) // N × 1
    }

    /// Same sphere but returned as a flat vector (N elements)
    fn fitness_1d(genes: &Array2<f64>, _context_id: usize) -> Array1<f64> {
        genes.map_axis(Axis(1), |ind| ind.iter().map(|&x| x * x).sum::<f64>())
    }

    /// Two objectives:
    ///   f₀ = Σ x²   |   f₁ = Σ |x|
    fn fitness_2d_two_obj(genes: &Array2<f64>, _context_id: usize) -> Array2<f64> {
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
    fn constraints_multi(genes: &Array2<f64>, _context_id: usize) -> Array2<f64> {
        let c0 = genes
            .sum_axis(Axis(1))
            .mapv(|s| s - 10.0)
            .insert_axis(Axis(1));
        let non_neg = genes.mapv(|x| -x);
        concatenate![Axis(1), c0, non_neg] // N × (d+1)
    }

    /// Single constraint: c = Σx – 10 ≤ 0
    fn constraints_single(genes: &Array2<f64>, _context_id: usize) -> Array1<f64> {
        genes.sum_axis(Axis(1)).mapv(|s| s - 10.0)
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 2-D fitness – no constraints
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn two_d_fitness_without_constraints_keeps_every_row() {
        let eval = EvaluatorBuilder::default()
            .fitness(fitness_2d_single)
            .constraints(NoConstraints)
            .keep_infeasible(true)
            .build()
            .expect("Builder failed");

        let genes = array![[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]];
        let fit = eval.evaluate(genes, 0).unwrap().fitness;

        // Each row keeps its sphere value in a single column
        let expected = array![[5.0], [25.0], [0.0]];
        assert_eq!(fit, expected);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 2-D fitness + constraints
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn multi_constraints_are_computed_correctly() {
        let eval = EvaluatorBuilder::default()
            .fitness(fitness_2d_single)
            .constraints(constraints_multi)
            .build()
            .expect("Builder failed");

        let genes = array![
            /* idx 0 */ [1.0, 2.0], // Σ=3  → c0 = -7 ; all genes ≥0
            /* idx 1 */ [3.0, 4.0], // Σ=7  → c0 = -3
            /* idx 2 */
            [5.0, 6.0], // Σ=11 → c0 =  1 (will be infeasible when filtering)
        ];
        let c = eval.evaluate(genes, 0).unwrap().constraints;

        let expected = array![[-7.0, -1.0, -2.0], [-3.0, -3.0, -4.0], [1.0, -5.0, -6.0],];
        assert_eq!(c, expected);
    }

    #[test]
    fn keep_infeasible_true_retains_every_row() {
        let eval = EvaluatorBuilder::default()
            .fitness(fitness_2d_single)
            .constraints(constraints_multi)
            .keep_infeasible(true)
            .build()
            .expect("Builder failed");

        let genes = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let pop = eval.evaluate(genes, 0).unwrap();
        assert_eq!(
            pop.genes.nrows(),
            3,
            "Nothing filtered when keep_infeasible = true"
        );
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 1-D fitness + multi-column constraints
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn one_d_fitness_multi_constraints_filters_by_constraint_only() {
        let eval = EvaluatorBuilder::default()
            .fitness(fitness_1d)
            .constraints(constraints_multi)
            .keep_infeasible(false)
            .build()
            .expect("Builder failed");

        let genes = array![
            /* 0 OK  */ [1.0, 2.0], // Σ=3  (feasible)
            /* 1 BAD */ [6.0, 5.0], // Σ=11 (violates c0)
        ];
        let pop = eval.evaluate(genes, 0).unwrap();

        assert_eq!(pop.genes.nrows(), 1, "Second row violates Σx - 10 ≤ 0");
        assert_eq!(pop.fitness, array![5.0]);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 1-D fitness + single constraint
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn one_d_fitness_single_constraint_filters_correctly() {
        let eval = EvaluatorBuilder::default()
            .fitness(fitness_1d)
            .constraints(constraints_single)
            .keep_infeasible(false)
            .build()
            .expect("Builder failed");

        let genes = array![
            /* 0 OK  */ [2.0, 3.0], // Σ=5  (feasible)
            /* 1 BAD */ [5.0, 6.0], // Σ=11 (violates)
        ];
        let pop = eval.evaluate(genes, 0).unwrap();

        assert_eq!(pop.genes.nrows(), 1);
        assert_eq!(pop.fitness, array![13.0]); // 2²+3²=13
        assert_eq!(
            pop.constraints,
            array![-5.0], // Σ - 10 = -5
            "Only constraint column for the surviving row"
        );
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 1-D fitness – everything filtered → EvaluatorError
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn all_rows_removed_returns_no_feasible_error() {
        let eval = EvaluatorBuilder::default()
            .fitness(fitness_1d)
            .constraints(constraints_multi)
            .keep_infeasible(false)
            .build()
            .expect("Builder failed");

        // Every candidate violates constraints (Σ>10)
        let genes = array![[6.0, 6.0], [5.5, 6.0], [6.0, 100.0]];
        let err = eval.evaluate(genes, 0).unwrap_err();
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
        let eval = EvaluatorBuilder::default()
            .fitness(fitness_2d_two_obj)
            .constraints(NoConstraints)
            .build()
            .expect("Builder failed");

        let genes = array![[1.0, 2.0], [3.0, 4.0]];
        let fit = eval.evaluate(genes, 0).unwrap().fitness;

        // Row-wise expected values: [Σx², Σ|x|]
        let expected = array![[5.0, 3.0], [25.0, 7.0]];
        assert_eq!(fit, expected);
    }
}
