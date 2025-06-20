use crate::algorithms::AlgorithmMOOBuilderError;

// Helper function for probability validation
pub(in crate::algorithms) fn validate_probability(
    value: f64,
    name: &str,
) -> Result<(), AlgorithmMOOBuilderError> {
    if !(0.0..=1.0).contains(&value) {
        return Err(AlgorithmMOOBuilderError::ValidationError(format!(
            "{} must be between 0 and 1, got {}",
            name, value
        )));
    }
    Ok(())
}

// Helper function for positive integer validation
pub(in crate::algorithms) fn validate_positive(
    value: usize,
    name: &str,
) -> Result<(), AlgorithmMOOBuilderError> {
    if value == 0 {
        return Err(AlgorithmMOOBuilderError::ValidationError(format!(
            "{} must be greater than 0",
            name
        )));
    }
    Ok(())
}

pub(in crate::algorithms) fn validate_bounds(
    lower_bound: Option<f64>,
    upper_bound: Option<f64>,
) -> Result<(), AlgorithmMOOBuilderError> {
    if let (Some(lower), Some(upper)) = (lower_bound, upper_bound) {
        if lower >= upper {
            return Err(AlgorithmMOOBuilderError::ValidationError(format!(
                "Lower bound ({}) must be less than upper bound ({})",
                lower, upper
            )));
        }
    }
    Ok(())
}
