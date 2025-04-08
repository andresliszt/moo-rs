use faer::linalg::solvers::Solve;
use faer::prelude::*;
use faer_ext::{IntoFaer, IntoNdarray};
use ndarray::{Array1, Array2, ArrayView1};

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

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
