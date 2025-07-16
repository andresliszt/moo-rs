//! # `genetic` – Core Data Structures
//!
//! This module defines the **fundamental types** that flow through every
//! evolutionary algorithm in *moors*—from initial sampling to final Pareto
//! archive.  They are intentionally *minimal* (pure `ndarray` wrappers) so they
//! can be inspected, cloned, or serialised without pulling extra dependencies.
use crate::private::{SealedD01, SealedD12};
use ndarray::{
    Array1, Array2, ArrayBase, ArrayView, ArrayView1, Axis, Dimension, Ix0, Ix1, Ix2, OwnedRepr,
    RemoveAxis, concatenate,
};
use num_traits::Zero;

pub type Constraints<D> = ArrayBase<OwnedRepr<f64>, D>;
pub type Fitness<D> = ArrayBase<OwnedRepr<f64>, D>;

pub trait D01: SealedD01 + Dimension {}

impl D01 for Ix0 {} // for Array0<T>  (scalar)
impl D01 for Ix1 {} // for Array1<T>  (vector)

/// Dimensions allowed in moors are 1D and 2D. Fitness might be 1D
/// (Single Objective Optimization) or 2D (Multi Objective Optimizations), in this
/// case each column represents an objective. In both, each row is an individual. We
/// restrict the implementors to this traits to ndarray::Ix1 and ndarray:Ix2 only.
pub trait D12: SealedD12 + Dimension + RemoveAxis {}

impl D12 for Ix1 {}
impl D12 for Ix2 {}

/// Represents an individual with genes, fitness, optional constraints,
/// rank, and an optional survival score.
#[derive(Debug, Clone)]
pub struct Individual<'a, FDim, ConstrDim>
where
    FDim: D01,
    ConstrDim: D01,
{
    pub genes: ArrayView1<'a, f64>,
    pub fitness: ArrayView<'a, f64, FDim>,
    pub constraints: ArrayView<'a, f64, ConstrDim>,
    pub rank: Option<usize>,
    pub survival_score: Option<f64>,
    pub constraint_violation_totals: Option<f64>,
}

impl<'a, FDim, ConstrDim> Individual<'a, FDim, ConstrDim>
where
    FDim: D01,
    ConstrDim: D01,
{
    /// Creates a new `Individual` with given genes, fitness, and constraints.
    /// `rank` and `survival_score` are initialized to `None`.
    pub fn new(
        genes: ArrayView1<'a, f64>,
        fitness: ArrayView<'a, f64, FDim>,
        constraints: ArrayView<'a, f64, ConstrDim>,
    ) -> Self {
        let constraint_violation_totals = match ConstrDim::NDIM {
            Some(0) => {
                let val = constraints.first().copied().unwrap_or(0.0);
                Some(if val <= 0.0 { 0.0 } else { val })
            }
            _ => {
                let sum = constraints.iter().copied().filter(|&v| v > 0.0).sum();
                Some(sum)
            }
        };
        Self {
            genes,
            fitness,
            constraints,
            rank: None,
            survival_score: None,
            constraint_violation_totals,
        }
    }

    /// Checks if the individual is feasible.
    pub fn is_feasible(&self) -> bool {
        match self.constraint_violation_totals {
            Some(val) => val == 0.0,
            None => true,
        }
    }

    /// Sets the rank of the individual.
    pub fn set_rank(&mut self, rank: usize) {
        self.rank = Some(rank);
    }

    /// Sets the survival score of the individual.
    pub fn set_survival_score(&mut self, survival_score: f64) {
        self.survival_score = Some(survival_score);
    }
}

impl<'a, FDim> Individual<'a, FDim, Ix1>
where
    FDim: D01,
{
    pub fn new_unconstrained(
        genes: ArrayView1<'a, f64>,
        fitness: ArrayView<'a, f64, FDim>,
    ) -> Self {
        Self {
            genes,
            fitness,
            constraints: ArrayView1::from(&[]),
            rank: None,
            survival_score: None,
            constraint_violation_totals: None,
        }
    }
}

/// The `Population` struct contains genes, fitness, constraints (if any),
/// rank (optional), and optionally a survival score vector.
#[derive(Debug, Clone)]
pub struct Population<FDim = Ix2, ConstrDim = Ix2>
where
    FDim: D12,
    ConstrDim: D12,
{
    pub genes: Array2<f64>,
    pub fitness: Fitness<FDim>,
    pub constraints: Constraints<ConstrDim>,
    pub rank: Option<Array1<usize>>,
    pub survival_score: Option<Array1<f64>>,
    pub constraint_violation_totals: Option<Array1<f64>>,
}

impl<FDim, ConstrDim> Population<FDim, ConstrDim>
where
    FDim: D12,
    ConstrDim: D12,
{
    const CONSTRAINTS_VIOLATION_TOLERANCE: f64 = 1e-6;

    /// Creates a new `Population` instance with the given genes, fitness, constraints, and rank.
    /// The `survival_score` field is set to `None` by default.
    pub fn new(
        genes: Array2<f64>,
        fitness: Fitness<FDim>,
        constraints: Constraints<ConstrDim>,
    ) -> Self {
        let constraint_violation = match ConstrDim::NDIM {
            Some(1) => {
                let tmp = constraints.mapv(|x| x.max(0.0));
                let mut arr = tmp.into_dimensionality::<Ix1>().unwrap();
                arr.mapv_inplace(|v| (v - Self::CONSTRAINTS_VIOLATION_TOLERANCE).max(0.0));
                Some(arr)
            }
            _ => {
                let tmp = constraints.mapv(|x| x.max(0.0)).sum_axis(Axis(1));
                let mut arr = tmp.into_dimensionality::<Ix1>().unwrap();
                arr.mapv_inplace(|v| (v - Self::CONSTRAINTS_VIOLATION_TOLERANCE).max(0.0));
                Some(arr)
            }
        };
        Self {
            genes,
            fitness,
            constraints,
            rank: None,
            survival_score: None,
            constraint_violation_totals: constraint_violation,
        }
    }

    pub fn get<'a>(
        &'a self,
        idx: usize,
    ) -> Individual<'a, <FDim as Dimension>::Smaller, <ConstrDim as Dimension>::Smaller>
    where
        <FDim as Dimension>::Smaller: D01,
        <ConstrDim as Dimension>::Smaller: D01,
    {
        let genes: ArrayView1<'a, f64> = self.genes.row(idx);
        let fitness = self.fitness.index_axis(Axis(0), idx);
        let constraints = self.constraints.index_axis(Axis(0), idx);

        let rank = self.rank.as_ref().map(|r| r[idx]);
        let survival_score = self.survival_score.as_ref().map(|s| s[idx]);
        let constraint_violation_totals =
            self.constraint_violation_totals.as_ref().map(|cv| cv[idx]);
        Individual {
            genes,
            fitness,
            constraints,
            rank,
            survival_score,
            constraint_violation_totals,
        }
    }
    /// Returns a new `Population` containing only the individuals at the specified indices.
    pub fn selected(&self, indices: &[usize]) -> Self {
        let genes = self.genes.select(Axis(0), indices);
        let fitness = self.fitness.select(Axis(0), indices);
        let constraints = self.constraints.select(Axis(0), indices);
        let rank = self.rank.as_ref().map(|r| r.select(Axis(0), indices));
        let constraint_violation_totals = self
            .constraint_violation_totals
            .as_ref()
            .map(|r| r.select(Axis(0), indices));
        let survival_score = self
            .survival_score
            .as_ref()
            .map(|ss| ss.select(Axis(0), indices));
        Population {
            genes,
            fitness,
            constraints,
            rank,
            survival_score,
            constraint_violation_totals,
        }
    }

    /// Returns the number of individuals in the population.
    pub fn len(&self) -> usize {
        self.genes.nrows()
    }

    pub fn is_empty(&self) -> bool {
        self.genes.nrows().is_zero()
    }

    /// Returns a new `Population` containing only the individuals with rank = 0.
    /// If no ranking information is available, the entire population is returned.
    pub fn best(&self) -> Self {
        if let Some(ranks) = &self.rank {
            let indices: Vec<usize> = ranks
                .iter()
                .enumerate()
                .filter_map(|(i, &r)| if r == 0 { Some(i) } else { None })
                .collect();
            self.selected(&indices)
        } else {
            // If rank is not set, return the entire population.
            self.clone()
        }
    }

    /// Updates the population's `survival_score` field.
    pub fn set_survival_score(&mut self, score: Array1<f64>) {
        self.survival_score = Some(score);
    }

    /// Updates the population's `rank` field.
    pub fn set_rank(&mut self, rank: Array1<usize>) {
        self.rank = Some(rank);
    }

    /// Merges two populations into one.
    pub fn merge(
        population1: &Population<FDim, ConstrDim>,
        population2: &Population<FDim, ConstrDim>,
    ) -> Population<FDim, ConstrDim> {
        // Concatenate genes (assumed to be an Array2).
        let merged_genes = concatenate(
            Axis(0),
            &[population1.genes.view(), population2.genes.view()],
        )
        .expect("Failed to merge genes");

        // Concatenate fitness (assumed to be an Array2).
        let merged_fitness = concatenate(
            Axis(0),
            &[population1.fitness.view(), population2.fitness.view()],
        )
        .expect("Failed to merge fitness");

        let merged_constraints = concatenate(
            Axis(0),
            &[
                population1.constraints.view(),
                population2.constraints.view(),
            ],
        )
        .expect("Failed to merge genes");

        // Merge rank: both must be Some or both must be None.
        let merged_rank = match (&population1.rank, &population2.rank) {
            (Some(r1), Some(r2)) => {
                Some(concatenate(Axis(0), &[r1.view(), r2.view()]).expect("Failed to merge rank"))
            }
            (None, None) => None,
            _ => panic!("Mismatched population rank: one is set and the other is None"),
        };

        let merged_total_cv = match (
            &population1.constraint_violation_totals,
            &population2.constraint_violation_totals,
        ) {
            (Some(r1), Some(r2)) => {
                Some(concatenate(Axis(0), &[r1.view(), r2.view()]).expect("Failed to merge rank"))
            }
            (None, None) => None,
            _ => panic!("Mismatched population rank: one is set and the other is None"),
        };

        // Merge survival_score: both must be Some or both must be None.
        let merged_survival_score = match (&population1.survival_score, &population2.survival_score)
        {
            (Some(s1), Some(s2)) => Some(
                concatenate(Axis(0), &[s1.view(), s2.view()])
                    .expect("Failed to merge survival scores"),
            ),
            (None, None) => None,
            _ => panic!("Mismatched population survival scores: one is set and the other is None"),
        };

        Population {
            genes: merged_genes,
            fitness: merged_fitness,
            constraints: merged_constraints,
            rank: merged_rank,
            survival_score: merged_survival_score,
            constraint_violation_totals: merged_total_cv,
        }
    }
}

impl<FDim> Population<FDim, Ix2>
where
    FDim: D12,
{
    pub fn new_unconstrained(genes: Array2<f64>, fitness: Fitness<FDim>) -> Self {
        let n = genes.nrows();
        Self {
            genes,
            fitness,
            constraints: Array2::zeros((n, 0)),
            rank: None,
            survival_score: None,
            constraint_violation_totals: None,
        }
    }
}

/// Type alias for Population in Multi Objective Optimization
pub type PopulationMOO<ConstrDim = Ix2> = Population<Ix2, ConstrDim>;
/// Type alias for Population in Single Objective Optimization
pub type PopulationSOO<ConstrDim = Ix1> = Population<Ix1, ConstrDim>;
/// Type alias for Individual in Multi Objective Optimization
pub type IndividualMOO<'a, ConstrDim> = Individual<'a, Ix1, ConstrDim>;
/// Type alias for Individual in Single Objective Optimization
pub type IndividualSOO<'a, ConstrDim> = Individual<'a, Ix0, ConstrDim>;
/// Type alias for a vector of `Population` representing multiple fronts.
pub type Fronts<ConstrDim> = Vec<PopulationMOO<ConstrDim>>;

/// An extension trait for `Fronts` that adds a `.to_population()` method
/// which flattens multiple fronts into a single `Population`.
pub(crate) trait FrontsExt<ConstrDim>
where
    ConstrDim: D12,
{
    fn to_population(self) -> PopulationMOO<ConstrDim>;
}

impl<ConstrDim> FrontsExt<ConstrDim> for Vec<PopulationMOO<ConstrDim>>
where
    ConstrDim: D12,
{
    fn to_population(self) -> PopulationMOO<ConstrDim> {
        self.into_iter()
            .reduce(|pop1, pop2| PopulationMOO::merge(&pop1, &pop2))
            .expect("Error when merging population vector")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr0, array};

    #[test]
    fn test_individual_moo_is_feasible() {
        // Individual with no constraints should be feasible.
        let genes_ind1 = array![1.0, 2.0];
        let fitness_ind1 = array![0.5, 1.0];
        let ind1 = IndividualMOO::new_unconstrained(genes_ind1.view(), fitness_ind1.view());
        assert!(
            ind1.is_feasible(),
            "Individual with no constraints should be feasible"
        );

        // Individual with constraints summing to <= 0 is feasible.
        let genes_ind2 = array![1.0, 2.0];
        let fitness_ind2 = array![0.5, 1.0];
        let constraints_ind2 = array![-1.0, 0.0];
        let ind2 = IndividualMOO::new(
            genes_ind2.view(),
            fitness_ind2.view(),
            constraints_ind2.view(),
        );
        assert!(
            ind2.is_feasible(),
            "Constraints sum -1.0 should be feasible"
        );

        // Individual with constraints summing to > 0 is not feasible.
        let genes_ind3 = array![1.0, 2.0];
        let fitness_ind3 = array![0.5, 1.0];
        let constraints_ind3 = array![1.0, 0.1];
        let ind3 = Individual::new(
            genes_ind3.view(),
            fitness_ind3.view(),
            constraints_ind3.view(),
        );
        assert!(
            !ind3.is_feasible(),
            "Constraints sum 1.1 should not be feasible"
        );
    }

    #[test]
    fn test_population_moo_new_get_selected_len() {
        // Create a population with two individuals.
        let genes = array![[1.0, 2.0], [3.0, 4.0]];
        let fitness = array![[0.5, 1.0], [1.5, 2.0]];
        // Using a rank array here.
        let rank = array![0, 1];
        let mut pop = PopulationMOO::new_unconstrained(genes.clone(), fitness.clone());
        pop.set_rank(rank);

        // Test len()
        assert_eq!(pop.len(), 2, "Population should have 2 individuals");

        // Test get()
        let ind0 = pop.get(0);
        assert_eq!(ind0.genes, genes.row(0).to_owned());
        assert_eq!(ind0.fitness, fitness.row(0).to_owned());
        assert_eq!(ind0.rank, Some(0));

        // Test selected()
        let selected = pop.selected(&[1]);
        assert_eq!(
            selected.len(),
            1,
            "Selected population should have 1 individual"
        );
        let ind_selected = selected.get(0);
        assert_eq!(ind_selected.genes, array![3.0, 4.0]);
        assert_eq!(ind_selected.fitness, array![1.5, 2.0]);
        assert_eq!(ind_selected.rank, Some(1));
    }

    #[test]
    fn test_population_moo_best_with_rank() {
        // Create a population with three individuals and varying ranks.
        let genes = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitness = array![[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]];
        // First and third individuals have rank 0, second has rank 1.
        let rank = array![0, 1, 0];
        let mut pop = PopulationMOO::new_unconstrained(genes, fitness);
        pop.set_rank(rank);
        let best = pop.best();
        // Expect best population to contain only individuals with rank 0.
        assert_eq!(best.len(), 2, "Best population should have 2 individuals");
        for i in 0..best.len() {
            let ind = best.get(i);
            assert_eq!(
                ind.rank,
                Some(0),
                "All individuals in best population should have rank 0"
            );
        }
    }

    #[test]
    fn test_population_moo_best_without_rank() {
        // Create a population without rank information.
        let genes = array![[1.0, 2.0], [3.0, 4.0]];
        let fitness = array![[0.5, 1.0], [1.5, 2.0]];
        let pop = PopulationMOO::new_unconstrained(genes.clone(), fitness.clone());
        // Since there is no rank, best() should return the whole population.
        let best = pop.best();
        assert_eq!(
            best.len(),
            pop.len(),
            "Best population should equal the original population when rank is None"
        );
    }

    #[test]
    fn test_set_survival_score() {
        // Create a population with two individuals.
        let genes = array![[1.0, 2.0], [3.0, 4.0]];
        let fitness = array![[0.5, 1.0], [1.5, 2.0]];
        let mut pop = PopulationMOO::new_unconstrained(genes, fitness);
        // Set a survival score vector with correct length.
        let score = array![0.1, 0.2];
        pop.set_survival_score(score.clone());
        assert_eq!(pop.survival_score.unwrap(), score);
    }

    #[test]
    fn test_population_moo_merge() {
        // Create two populations with rank information.
        let genes1 = array![[1.0, 2.0], [3.0, 4.0]];
        let fitness1 = array![[0.5, 1.0], [1.5, 2.0]];
        let rank1 = array![0, 0];
        let mut pop1 = PopulationMOO::new_unconstrained(genes1, fitness1);
        pop1.set_rank(rank1);

        let genes2 = array![[5.0, 6.0], [7.0, 8.0]];
        let fitness2 = array![[2.5, 3.0], [3.5, 4.0]];
        let rank2 = array![1, 1];
        let mut pop2 = PopulationMOO::new_unconstrained(genes2, fitness2);
        pop2.set_rank(rank2);

        let merged = PopulationMOO::merge(&pop1, &pop2);
        assert_eq!(
            merged.len(),
            4,
            "Merged population should have 4 individuals"
        );

        let expected_genes = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        assert_eq!(merged.genes, expected_genes, "Merged genes do not match");

        let expected_fitness = array![[0.5, 1.0], [1.5, 2.0], [2.5, 3.0], [3.5, 4.0]];
        assert_eq!(
            merged.fitness, expected_fitness,
            "Merged fitness does not match"
        );

        let expected_rank = Some(array![0, 0, 1, 1]);
        assert_eq!(merged.rank, expected_rank, "Merged rank does not match");
    }

    #[test]
    fn test_fronts_ext_to_population_moo() {
        // Create two fronts.
        let genes1 = array![[1.0, 2.0], [3.0, 4.0]];
        let fitness1 = array![[0.5, 1.0], [1.5, 2.0]];
        let pop1 = PopulationMOO::new_unconstrained(genes1, fitness1);

        let genes2 = array![[5.0, 6.0], [7.0, 8.0]];
        let fitness2 = array![[2.5, 3.0], [3.5, 4.0]];
        let pop2 = PopulationMOO::new_unconstrained(genes2, fitness2);

        let fronts = vec![pop1.clone(), pop2.clone()];
        let merged = fronts.to_population();

        assert_eq!(
            merged.len(),
            4,
            "Flattened population should have 4 individuals"
        );

        let expected_genes = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        assert_eq!(merged.genes, expected_genes, "Flattened genes do not match");
    }

    #[test]
    #[should_panic(
        expected = "Mismatched population survival scores: one is set and the other is None"
    )]
    fn test_population_moo_merge_mismatched_survival_score() {
        let genes1 = array![[1.0, 2.0]];
        let fitness1 = array![[0.5, 1.0]];
        let mut pop1 = PopulationMOO::new_unconstrained(genes1, fitness1);
        let score1 = array![0.1];
        pop1.set_survival_score(score1);

        let genes2 = array![[3.0, 4.0]];
        let fitness2 = array![[1.5, 2.0]];
        let pop2 = PopulationMOO::new_unconstrained(genes2, fitness2);

        Population::merge(&pop1, &pop2);
    }

    #[test]
    fn test_individual_soo_with_and_without_constraints() {
        // Unconstrained individual: fitness is 0-D
        let genes = array![0.1, 0.2, 0.3];
        let fitness = arr0(42.0);
        let ind_unconstrained = IndividualSOO::new_unconstrained(genes.view(), fitness.view());
        // Should have no constraints
        assert_eq!(ind_unconstrained.constraints, ArrayView1::from(&[]));
        assert!(ind_unconstrained.is_feasible());
        assert_eq!(ind_unconstrained.rank, None);
        assert_eq!(ind_unconstrained.survival_score, None);

        // Constrained individual with a scalar constraint ≤ 0
        let constraint_ok = arr0(-0.5);
        let ind_ok = IndividualSOO::new(genes.view(), fitness.view(), constraint_ok.view());
        let c_ok = ind_ok.constraints.into_scalar();
        assert_eq!(*c_ok, -0.5);
        assert!(ind_ok.is_feasible());

        // Constrained individual with a scalar constraint > 0
        let constraint_fail = arr0(1.5);
        let ind_fail = IndividualSOO::new(genes.view(), fitness.view(), constraint_fail.view());
        let c_fail = ind_fail.constraints.into_scalar();
        assert_eq!(*c_fail, 1.5);
        assert!(!ind_fail.is_feasible());
    }
}
