use faer::prelude::*;
use faer_ext::{IntoFaer, IntoNdarray};
use faer::linalg::solvers::Solve;
use ndarray::{Array1, Array2, ArrayView1};
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::genetic::PopulationFitness;
use crate::helpers::extreme_points::get_nadir;

pub trait HyperPlaneNormalization {
    /// This corresponds to the Z_max defined in the NSGA3 - AGEMOEA referenced papers
    fn compute_extreme_points(&self, population_fitness: &PopulationFitness) -> Array2<f64>;

    /// Computes the intercepts vector `a` by solving the linear system:
    /// Z_max * b = 1, where 1 is a vector of ones.
    /// then the intercepts in the objective axis are given by a = 1/b
    fn compute_hyperplane_intercepts(&self, population_fitness: &PopulationFitness) -> Array1<f64> {
        let m = population_fitness.ncols();
        // Compute Z_max
        let z_max = self.compute_extreme_points(&population_fitness);
        // We have to use faer  solver --- We don't use ndarray-linalg due that is not maintained
        let z_max_faer = z_max.view().into_faer();

        let ones = Mat::<f64>::from_fn(m, 1, |_, _| 1.0);
        // Compute the LU decomposition with partial pivoting,
        let plu = z_max_faer.partial_piv_lu();
        let solution = plu.solve(&ones);
        let solution_ndarray = solution.as_ref().into_ndarray();
        // this step is done because faer responds as two array [[...], [...], ..., [...]]
        let solution_ndarray: ArrayView1<f64> = solution_ndarray
            .into_shape_with_order(solution_ndarray.len())
            .unwrap();
        if solution_ndarray.iter().any(|&x| !x.is_finite() || x <= 0.0) {
            // this is the case for singullar matrices
            get_nadir(&population_fitness)
        } else {
            // Calculate intercepts as 1 / a.
            let intercept = solution_ndarray.mapv(|val| 1.0 / val);
            // Additional check: if the computed intercept is less than the observed maximum,
            // use the observed maximum (fallback to min-max).
            let fallback = get_nadir(&population_fitness);
            let combined: Vec<f64> = intercept
                .iter()
                .zip(fallback.iter())
                .map(|(&calc, &fb)| if calc < fb { fb } else { calc })
                .collect();
            Array1::from(combined)
        }
    }
}

/// Returns the smallest value of H such that the number of Das-Dennis reference points
/// (computed as binom(H + m - 1, m - 1)) is greater than or equal to `n_reference_points`.
fn choose_h(n_reference_points: usize, n_objectives: usize) -> usize {
    let mut h = 1;
    loop {
        let n_points = binomial_coefficient(h + n_objectives - 1, n_objectives - 1);
        if n_points >= n_reference_points {
            return h;
        }
        h += 1;
    }
}

/// Computes the binomial coefficient "n choose k".
fn binomial_coefficient(n: usize, k: usize) -> usize {
    let mut result = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Recursively generates all combinations of nonnegative integers of length `m` that sum to `sum`.
///
/// - `n_objectives`: total number of objectives
/// - `sum`: the remaining sum to distribute among the components
/// - `index`: current index being filled
/// - `current`: holds the current combination under construction
/// - `points`: collects all generated combinations
fn generate_combinations(
    n_objectives: usize,
    sum: usize,
    index: usize,
    current: &mut Vec<usize>,
    points: &mut Vec<Vec<usize>>,
) {
    if index == n_objectives - 1 {
        // For the last component, assign the remaining sum.
        current.push(sum);
        points.push(current.clone());
        current.pop();
        return;
    }
    // Distribute values from 0 to `sum` for the current component.
    for x in 0..=sum {
        current.push(x);
        generate_combinations(n_objectives, sum - x, index + 1, current, points);
        current.pop();
    }
}

/// A common trait for structured reference points.
pub trait StructuredReferencePoints {
    fn generate(&self) -> Array2<f64>;
}

#[derive(Clone, Debug)]
pub struct DanAndDenisReferencePoints {
    n_reference_points: usize,
    n_objectives: usize,
}

impl StructuredReferencePoints for DanAndDenisReferencePoints {
    /// Generates all Das-Dennis reference points given a population size and number of objectives.
    ///
    /// The procedure is:
    /// 1. Estimate H using `choose_h(population_size, m)`.
    /// 2. Generate all combinations of nonnegative integers (h₁, h₂, …, hₘ) that satisfy:
    ///      h₁ + h₂ + ... + hₘ = H.
    /// 3. Normalize each combination by dividing each component by H to get a point on the simplex.
    ///
    /// The function returns an Array2<f64> where each row is a reference point.
    fn generate(&self) -> Array2<f64> {
        // Step 1: Estimate H using the population size and number of objectives.
        let h = choose_h(self.n_reference_points, self.n_objectives);

        // Step 2: Generate all combinations (h₁, h₂, …, hₘ) such that h₁ + h₂ + ... + hₘ = H.
        let mut points: Vec<Vec<usize>> = Vec::new();
        let mut current: Vec<usize> = Vec::with_capacity(self.n_objectives);
        generate_combinations(self.n_objectives, h, 0, &mut current, &mut points);

        // Step 3: Normalize each combination by dividing by H and store in an Array2.
        let num_points = points.len();
        let mut arr = Array2::<f64>::zeros((num_points, self.n_objectives));
        for (i, combination) in points.iter().enumerate() {
            for j in 0..self.n_objectives {
                arr[[i, j]] = combination[j] as f64 / h as f64;
            }
        }
        arr
    }
}

/// An enum that can hold any supported structured reference points type.
pub enum PyStructuredReferencePointsEnum {
    DanAndDenis(PyDanAndDenisReferencePoints),
}

/// Implement extraction for the enum.
impl<'py> FromPyObject<'py> for PyStructuredReferencePointsEnum {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(dan) = ob.extract::<PyDanAndDenisReferencePoints>() {
            Ok(PyStructuredReferencePointsEnum::DanAndDenis(dan))
        } else {
            Err(PyTypeError::new_err(
                "reference_points must be one of the supported structured reference point types.",
            ))
        }
    }
}

/// Implement the trait for the enum.
impl PyStructuredReferencePointsEnum {
    pub fn generate(&self) -> Array2<f64> {
        match self {
            PyStructuredReferencePointsEnum::DanAndDenis(d) => d.inner.generate(),
        }
    }
}

/// Expose the DanAndDenisReferencePoints struct to Python.
#[pyclass(name = "DanAndDenisReferencePoints")]
#[derive(Clone, Debug)]
pub struct PyDanAndDenisReferencePoints {
    pub inner: DanAndDenisReferencePoints,
}

#[pymethods]
impl PyDanAndDenisReferencePoints {
    #[new]
    pub fn new(n_reference_points: usize, n_objectives: usize) -> Self {
        PyDanAndDenisReferencePoints {
            inner: DanAndDenisReferencePoints {
                n_reference_points,
                n_objectives,
            },
        }
    }
    /// Returns the reference points as a NumPy array.
    fn generate<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let array = self.inner.generate();
        array.into_pyarray(py)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    // Test implementation for non-singular z_max.
    struct TestHyperPlaneNormalizerNonSingular;

    impl HyperPlaneNormalization for TestHyperPlaneNormalizerNonSingular {
        fn compute_extreme_points(&self, _population_fitness: &PopulationFitness) -> Array2<f64> {
            // Return a non-singular (diagonal) matrix:
            // [ [2.0, 0.0],
            //   [0.0, 0.5] ]
            Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 0.5]).unwrap()
        }
    }

    // Test implementation for singular z_max.
    struct TestHyperPlaneNormalizerSingular;

    impl HyperPlaneNormalization for TestHyperPlaneNormalizerSingular {
        fn compute_extreme_points(&self, _population_fitness: &PopulationFitness) -> Array2<f64> {
            // Return a singular matrix:
            // [ [1.0, 2.0],
            //   [2.0, 4.0] ]
            Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 4.0]).unwrap()
        }
    }

    #[test]
    fn test_compute_hyperplane_intercepts_non_singular() {
        // Non-singular case:
        // Define the population fitness as:
        // pop_fit = [ [0.2, 0.3],
        //             [1.0, 1.0],
        //             [0.9, 0.8] ]
        // Then, get_nadir(pop_fit) computes the maximum of each column: [1.0, 1.0].
        // With z_max from TestHyperPlaneNormalizerNonSingular,
        // the system Z_max * b = [1, 1]^T yields b = [0.5, 2.0] and intercepts = [2.0, 0.5].
        // Finally, combining element-wise with the fallback:
        //   max(2.0, 1.0) = 2.0 and max(0.5, 1.0) = 1.0.
        // Expected result: [2.0, 1.0].

        let pop_fit = array![[0.2, 0.3], [1.0, 1.0], [0.9, 0.8]];

        let normalizer = TestHyperPlaneNormalizerNonSingular;
        let result = normalizer.compute_hyperplane_intercepts(&pop_fit);
        let expected = array![2.0, 1.0];

        assert_eq!(
            result, expected,
            "Non-singular test failed: expected {:?}, got {:?}",
            expected, result
        );
    }

    #[test]
    fn test_compute_hyperplane_intercepts_singular() {
        // Singular case:
        // Define the population fitness as:
        // pop_fit = [ [5.0, 6.0],
        //             [4.0, 5.0] ]
        // Then, get_nadir(pop_fit) returns [5.0, 6.0].
        // With z_max from TestHyperPlaneNormalizerSingular (a singular matrix),
        // the LU solver produces non-finite values so the function returns the fallback.
        // Expected result: [5.0, 6.0].

        let pop_fit = array![[5.0, 6.0], [4.0, 5.0]];

        let normalizer = TestHyperPlaneNormalizerSingular;
        let result = normalizer.compute_hyperplane_intercepts(&pop_fit);
        let expected = array![5.0, 6.0];

        assert_eq!(
            result, expected,
            "Singular test failed: expected {:?}, got {:?}",
            expected, result
        );
    }
}
