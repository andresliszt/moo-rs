use crate::genetic::{Population, PopulationConstraints, PopulationFitness, PopulationGenes};
use ndarray::Axis;
use std::fmt;

/// Error type for the Evaluator.
#[derive(Debug)]
pub enum EvaluatorError {
    NoFeasibleIndividuals,
}

impl fmt::Display for EvaluatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvaluatorError::NoFeasibleIndividuals => {
                write!(f, "No feasible individuals found in the population.")
            }
        }
    }
}

/// Evaluator struct for calculating fitness and (optionally) constraints,
/// then assembling a `Population`. In addition to the user-provided constraints function,
/// optional lower and upper bounds can be specified for the decision variables (genes).
pub struct Evaluator<F, G>
where
    F: Fn(&PopulationGenes) -> PopulationFitness,
    G: Fn(&PopulationGenes) -> PopulationConstraints,
{
    fitness_fn: F,
    constraints_fn: Option<G>,
    keep_infeasible: bool,
    /// Optional lower bound for each gene.
    lower_bound: Option<f64>,
    /// Optional upper bound for each gene.
    upper_bound: Option<f64>,
}

impl<F, G> Evaluator<F, G>
where
    F: Fn(&PopulationGenes) -> PopulationFitness,
    G: Fn(&PopulationGenes) -> PopulationConstraints,
{
    /// Creates a new `Evaluator` with a fitness function, an optional constraints function,
    /// a flag to keep infeasible individuals, and optional lower/upper bounds.
    pub fn new(
        fitness_fn: F,
        constraints_fn: Option<G>,
        keep_infeasible: bool,
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
    ) -> Self {
        Self {
            fitness_fn,
            constraints_fn,
            keep_infeasible,
            lower_bound,
            upper_bound,
        }
    }

    /// Evaluates the fitness of the population genes (2D ndarray).
    fn evaluate_fitness(&self, population_genes: &PopulationGenes) -> PopulationFitness {
        (self.fitness_fn)(population_genes)
    }

    /// Evaluates the constraints (if available).
    /// Returns `None` if no constraints function was provided.
    fn evaluate_constraints(
        &self,
        population_genes: &PopulationGenes,
    ) -> Option<PopulationConstraints> {
        self.constraints_fn.as_ref().map(|cf| cf(population_genes))
    }

    /// Builds the population instance from the genes. If `keep_infeasible` is false,
    /// individuals are filtered out if they do not satisfy:
    ///   - The provided constraints function (all constraint values must be ≤ 0), and
    ///   - The optional lower and upper bounds (each gene must satisfy lower_bound <= gene <= upper_bound).
    pub fn evaluate(&self, mut genes: PopulationGenes) -> Result<Population, EvaluatorError> {
        let mut fitness = self.evaluate_fitness(&genes);
        let mut constraints = self.evaluate_constraints(&genes);

        if !self.keep_infeasible {
            // Create a list of all indices.
            let n = genes.nrows();
            let mut feasible_indices: Vec<usize> = (0..n).collect();

            // Filter individuals that do not satisfy the constraints function (if provided).
            if let Some(ref c) = constraints {
                feasible_indices = feasible_indices
                    .into_iter()
                    .filter(|&i| c.index_axis(Axis(0), i).iter().all(|&val| val <= 0.0))
                    .collect();
            }

            // Further filter individuals based on the optional lower and upper bounds.
            if self.lower_bound.is_some() || self.upper_bound.is_some() {
                feasible_indices = feasible_indices
                    .into_iter()
                    .filter(|&i| {
                        let individual = genes.index_axis(Axis(0), i);
                        let lower_ok = self
                            .lower_bound
                            .map_or(true, |lb| individual.iter().all(|&x| x >= lb));
                        let upper_ok = self
                            .upper_bound
                            .map_or(true, |ub| individual.iter().all(|&x| x <= ub));
                        lower_ok && upper_ok
                    })
                    .collect();
            }

            // Filter all relevant arrays (genes, fitness, and constraints if present).
            genes = genes.select(Axis(0), &feasible_indices);

            if genes.nrows() == 0 {
                return Err(EvaluatorError::NoFeasibleIndividuals);
            }

            fitness = fitness.select(Axis(0), &feasible_indices);
            constraints = constraints.map(|c_array| c_array.select(Axis(0), &feasible_indices));
        }
        Ok(Population::new(genes, fitness, constraints, None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Axis;
    use ndarray::{array, concatenate};

    // Fitness function: Sphere function (sum of squares for each individual).
    fn fitness_fn(genes: &PopulationGenes) -> PopulationFitness {
        genes
            .map_axis(Axis(1), |individual| {
                individual.iter().map(|&x| x * x).sum::<f64>()
            })
            .insert_axis(Axis(1))
    }

    // Constraints function:
    //   Constraint 1: sum of genes - 10 ≤ 0.
    //   Constraint 2: each gene ≥ 0 (represented as -x ≤ 0).
    fn constraints_fn(genes: &PopulationGenes) -> PopulationConstraints {
        let sum_constraint = genes
            .sum_axis(Axis(1))
            .mapv(|sum| sum - 10.0)
            .insert_axis(Axis(1));
        let non_neg_constraints = genes.mapv(|x| -x);
        concatenate(
            Axis(1),
            &[sum_constraint.view(), non_neg_constraints.view()],
        )
        .unwrap()
    }

    #[test]
    fn test_evaluator_evaluate_fitness() {
        // Note: We explicitly specify the function types for the generic evaluator.
        let evaluator = Evaluator::new(
            fitness_fn,
            None::<fn(&PopulationGenes) -> PopulationConstraints>,
            true,
            None,
            None,
        );

        let population_genes = array![[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]];
        let fitness = evaluator.evaluate_fitness(&population_genes);
        let expected_fitness = array![
            [5.0],  // 1^2 + 2^2 = 5
            [25.0], // 3^2 + 4^2 = 25
            [0.0],  // 0^2 + 0^2 = 0
        ];

        assert_eq!(fitness, expected_fitness);
    }

    #[test]
    fn test_evaluator_evaluate_constraints() {
        let evaluator = Evaluator::new(fitness_fn, Some(constraints_fn), true, None, None);

        let population_genes = array![
            [1.0, 2.0], // Feasible (sum = 3, 3 - 10 = -7; genes ≥ 0)
            [3.0, 4.0], // Feasible (sum = 7, 7 - 10 = -3; genes ≥ 0)
            [5.0, 6.0]  // Infeasible (sum = 11, 11 - 10 = 1 > 0)
        ];

        if let Some(constraints_array) = evaluator.evaluate_constraints(&population_genes) {
            let expected_constraints = array![
                // Each row: [sum - 10, -gene1, -gene2]
                [-7.0, -1.0, -2.0],
                [-3.0, -3.0, -4.0],
                [1.0, -5.0, -6.0],
            ];
            assert_eq!(constraints_array, expected_constraints);

            let feasibility: Vec<bool> = constraints_array
                .outer_iter()
                .map(|row| row.iter().all(|&val| val <= 0.0))
                .collect();
            let expected_feasibility = vec![true, true, false];
            assert_eq!(feasibility, expected_feasibility);
        } else {
            panic!("The constraints function should not be None");
        }
    }

    #[test]
    fn test_evaluator_evaluate_without_bounds() {
        let evaluator = Evaluator::new(
            fitness_fn,
            Some(constraints_fn),
            true, // keep_infeasible = true, no filtering
            None,
            None,
        );

        let population_genes = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let population = evaluator.evaluate(population_genes).unwrap();

        // Verify that the total number of individuals is 3.
        assert_eq!(
            population.genes.nrows(),
            3,
            "Total number of individuals should be 3 without filtering."
        );

        if let Some(ref constraints) = population.constraints {
            assert_eq!(
                constraints.nrows(),
                3,
                "The number of rows in constraints should match the number of individuals."
            );
        }
        // In evaluate, rank is None.
        assert!(population.rank.is_none());
    }

    #[test]
    fn test_evaluator_evaluate_with_infeasible_and_bounds() {
        // Use the constraints function and bounds, with keep_infeasible = false.
        // The bounds require each gene to be between 0 and 5.
        let evaluator = Evaluator::new(
            fitness_fn,
            Some(constraints_fn),
            false,
            Some(0.0),
            Some(5.0),
        );

        // Create a population with 3 individuals:
        //   - [1.0, 2.0]: Feasible (sum = 3, genes between 0 and 5)
        //   - [3.0, 4.0]: Feasible (sum = 7, genes between 0 and 5)
        //   - [6.0, 1.0]: Infeasible (6.0 > 5.0)
        let population_genes = array![[1.0, 2.0], [3.0, 4.0], [6.0, 1.0]];
        let population = evaluator.evaluate(population_genes).unwrap();

        // Expecting only 2 feasible individuals after filtering.
        assert_eq!(
            population.genes.nrows(),
            2,
            "Total number of individuals should be 2 after filtering by feasibility and bounds."
        );
    }
}
