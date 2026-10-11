//! Python bindings for the `moors::metrics` quality indicators.
use moors::{generational_distance, hypervolume, inverted_generational_distance, spacing, spread};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::py_error::InvalidParameterError;

fn check_columns_match(front_cols: usize, other_cols: usize, other_name: &str) -> PyResult<()> {
    if front_cols != other_cols {
        return Err(InvalidParameterError::new_err(format!(
            "front has {front_cols} objectives but {other_name} has {other_cols}"
        )));
    }
    Ok(())
}

/// Hypervolume (HV) of `front` with respect to `reference` (minimization).
#[pyfunction]
#[pyo3(name = "hypervolume")]
pub fn py_hypervolume(
    front: PyReadonlyArray2<'_, f64>,
    reference: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let front = front.as_array().to_owned();
    let reference = reference.as_array().to_owned();
    check_columns_match(front.ncols(), reference.len(), "reference")?;
    Ok(hypervolume(&front, &reference))
}

/// Generational Distance (GD) of `front` with respect to `reference_front`.
#[pyfunction]
#[pyo3(name = "generational_distance")]
pub fn py_generational_distance(
    front: PyReadonlyArray2<'_, f64>,
    reference_front: PyReadonlyArray2<'_, f64>,
) -> PyResult<f64> {
    let front = front.as_array().to_owned();
    let reference_front = reference_front.as_array().to_owned();
    check_columns_match(front.ncols(), reference_front.ncols(), "reference_front")?;
    Ok(generational_distance(&front, &reference_front))
}

/// Inverted Generational Distance (IGD) of `front` with respect to `reference_front`.
#[pyfunction]
#[pyo3(name = "inverted_generational_distance")]
pub fn py_inverted_generational_distance(
    front: PyReadonlyArray2<'_, f64>,
    reference_front: PyReadonlyArray2<'_, f64>,
) -> PyResult<f64> {
    let front = front.as_array().to_owned();
    let reference_front = reference_front.as_array().to_owned();
    check_columns_match(front.ncols(), reference_front.ncols(), "reference_front")?;
    Ok(inverted_generational_distance(&front, &reference_front))
}

/// Spacing (SP) of `front`.
#[pyfunction]
#[pyo3(name = "spacing")]
pub fn py_spacing(front: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
    let front = front.as_array().to_owned();
    if front.nrows() < 2 {
        return Err(InvalidParameterError::new_err(
            "spacing requires at least 2 points in front",
        ));
    }
    Ok(spacing(&front))
}

/// Spread (Δ) of `front` with respect to `reference_front`. Only defined for 2 objectives.
#[pyfunction]
#[pyo3(name = "spread")]
pub fn py_spread(
    front: PyReadonlyArray2<'_, f64>,
    reference_front: PyReadonlyArray2<'_, f64>,
) -> PyResult<f64> {
    let front = front.as_array().to_owned();
    let reference_front = reference_front.as_array().to_owned();
    if front.ncols() != 2 || reference_front.ncols() != 2 {
        return Err(InvalidParameterError::new_err(
            "spread (Δ) is only defined for 2-objective fronts",
        ));
    }
    if front.nrows() < 2 {
        return Err(InvalidParameterError::new_err(
            "spread requires at least 2 points in front",
        ));
    }
    Ok(spread(&front, &reference_front))
}
