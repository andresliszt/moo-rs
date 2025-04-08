use faer::{Mat, MatRef};
use faer_ext::IntoFaer;

use ndarray::{Array2, ArrayView1, Axis};

/// Computes the Lp norm of a array view vector.
pub fn lp_norm_arrayview(x: &ArrayView1<f64>, p: f64) -> f64 {
    x.iter()
        .map(|&val| val.abs().powf(p))
        .sum::<f64>()
        .powf(1.0 / p)
}

fn faer_dot(x: MatRef<f64>, y: MatRef<f64>) -> Mat<f64> {
    x * y.transpose()
}

pub fn faer_dot_from_array(x: &Array2<f64>, y: &Array2<f64>) -> Mat<f64> {
    let faer_x = x.view().into_faer();
    let faer_y = y.view().into_faer();
    faer_dot(faer_x, faer_y)
}

pub fn faer_squared_norm(x: MatRef<f64>) -> Mat<f64> {
    let x_norm = Mat::from_fn(x.nrows(), 1, |i, _| {
        let row = x.row(i);
        row * row.transpose()
    });
    x_norm
}

pub fn faer_dot_and_norms(x: &Array2<f64>, y: &Array2<f64>) -> (Mat<f64>, Mat<f64>, Mat<f64>) {
    let faer_x = x.view().into_faer();
    let faer_y = y.view().into_faer();

    let x_norm = faer_squared_norm(faer_x);
    let y_norm = faer_squared_norm(faer_y);

    let faer_dot = faer_dot(faer_x, faer_y);
    (x_norm, y_norm, faer_dot)
}

/// Computes the cross squared Euclidean distance matrix between `data` and `reference`
/// using matrix algebra.
///
/// For data of shape (n, d) and reference of shape (m, d), returns an (n x m) matrix
/// where the (i,j) element is the squared Euclidean distance between the i-th row of data
/// and the j-th row of reference.
pub fn cross_euclidean_distances(data: &Array2<f64>, reference: &Array2<f64>) -> Mat<f64> {
    let (data_norm, reference_norm, faer_dot) = faer_dot_and_norms(data, reference);

    let n = data.nrows();
    let m = reference.nrows();
    // d²(i,j) = ||x_i||² + ||y_j||² - 2 * (x_i dot y_j)
    let faer_dist: Mat<f64> = Mat::from_fn(n, m, |i, j| {
        data_norm.get(i, 0) + reference_norm.get(j, 0) - 2.0 * faer_dot.get(i, j)
    });
    faer_dist
}

pub fn cross_p_distances(data: &Array2<f64>, reference: &Array2<f64>, p: f64) -> Array2<f64> {
    // Expand dimensions so that `data` has shape (n, 1, d) and `reference` has shape (1, m, d)
    // Expand dimensions and convert to owned arrays so that subtraction is allowed.
    let data_expanded = data.view().insert_axis(Axis(1)).to_owned(); // shape (n, 1, d)
    let reference_expanded = reference.view().insert_axis(Axis(0)).to_owned(); // shape (1, m, d)

    // Compute the element-wise differences.
    let diff = data_expanded - reference_expanded;

    // Compute the sum of |x - y|^p along the feature dimension (axis 2)
    let dists_p = diff.mapv(|x| x.abs().powf(p)).sum_axis(Axis(2));
    dists_p
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer_ext::IntoNdarray;
    use ndarray::array;

    #[test]
    fn test_cross_euclidean_distances() {
        // Create sample data and reference arrays (each row is a point in 2D space).
        let data = array![[0.0, 0.0], [1.0, 1.0]];
        let reference = array![[0.0, 0.0], [2.0, 2.0]];

        // Expected squared Euclidean distances:
        // - Distance between [0,0] and [0,0]: 0²+0² = 0
        // - Distance between [0,0] and [2,2]: 2²+2² = 8
        // - Distance between [1,1] and [0,0]: 1²+1² = 2
        // - Distance between [1,1] and [2,2]: 1²+1² = 2
        let expected = array![[0.0, 8.0], [2.0, 2.0]];

        let result = cross_euclidean_distances(&data, &reference);
        let result_ndarray = result.as_ref().into_ndarray();
        assert_eq!(result_ndarray, expected);
    }

    #[test]
    fn test_cross_p_distances() {
        // Create sample data and reference arrays.
        let data = array![[0.0, 0.0], [1.0, 1.0]];
        let reference = array![[0.0, 0.0], [2.0, 2.0]];

        // For p = 2, the function should return the sum of squared differences,
        // which is equivalent to the squared Euclidean distances.
        let expected_p2 = array![[0.0, 8.0], [2.0, 2.0]];
        let result_p2 = cross_p_distances(&data, &reference, 2.0);
        assert_eq!(result_p2, expected_p2);

        // For p = 1, the function should return the Manhattan distances (without taking any root).
        // Manhattan distances:
        // - [0,0] vs [0,0]: |0-0| + |0-0| = 0
        // - [0,0] vs [2,2]: |0-2| + |0-2| = 4
        // - [1,1] vs [0,0]: |1-0| + |1-0| = 2
        // - [1,1] vs [2,2]: |1-2| + |1-2| = 2
        let expected_p1 = array![[0.0, 4.0], [2.0, 2.0]];
        let result_p1 = cross_p_distances(&data, &reference, 1.0);
        assert_eq!(result_p1, expected_p1);
    }
}
