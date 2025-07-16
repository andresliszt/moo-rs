use crate::algorithms::AlgorithmBuilderError;

// Helper function for probability validation
pub(in crate::algorithms) fn validate_probability(
    value: f64,
    name: &str,
) -> Result<(), AlgorithmBuilderError> {
    if !(0.0..=1.0).contains(&value) {
        return Err(AlgorithmBuilderError::ValidationError(format!(
            "{name} must be between 0 and 1, got {value}"
        )));
    }
    Ok(())
}

// Helper function for positive integer validation
pub(in crate::algorithms) fn validate_positive(
    value: usize,
    name: &str,
) -> Result<(), AlgorithmBuilderError> {
    if value == 0 {
        return Err(AlgorithmBuilderError::ValidationError(format!(
            "{name} must be greater than 0"
        )));
    }
    Ok(())
}

pub(in crate::algorithms) fn validate_bounds(
    lower_bound: f64,
    upper_bound: f64,
) -> Result<(), AlgorithmBuilderError> {
    if lower_bound >= upper_bound {
        return Err(AlgorithmBuilderError::ValidationError(format!(
            "Lower bound ({lower_bound}) must be less than upper bound ({upper_bound})"
        )));
    }
    Ok(())
}
