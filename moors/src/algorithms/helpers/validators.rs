use crate::algorithms::helpers::error::MultiObjectiveAlgorithmError;

// Helper function for probability validation
pub(in crate::algorithms) fn validate_probability(
    value: f64,
    name: &str,
) -> Result<(), MultiObjectiveAlgorithmError> {
    if !(0.0..=1.0).contains(&value) {
        return Err(MultiObjectiveAlgorithmError::InvalidParameter(format!(
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
) -> Result<(), MultiObjectiveAlgorithmError> {
    if value == 0 {
        return Err(MultiObjectiveAlgorithmError::InvalidParameter(format!(
            "{} must be greater than 0",
            name
        )));
    }
    Ok(())
}

pub(in crate::algorithms) fn validate_bounds(
    lower_bound: Option<f64>,
    upper_bound: Option<f64>,
) -> Result<(), MultiObjectiveAlgorithmError> {
    if let (Some(lower), Some(upper)) = (lower_bound, upper_bound) {
        if lower >= upper {
            return Err(MultiObjectiveAlgorithmError::InvalidParameter(format!(
                "Lower bound ({}) must be less than upper bound ({})",
                lower, upper
            )));
        }
    }
    Ok(())
}
