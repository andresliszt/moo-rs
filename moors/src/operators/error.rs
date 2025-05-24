use thiserror::Error;

/* -------------------------------------------------------------------------- */
/*  Operator-specific errors                                                  */
/* -------------------------------------------------------------------------- */

/// Errors that can occur inside a mutation operator.
#[derive(Debug, Error)]
pub enum MutationError {
    #[error("value must be positive; received {0}")]
    PositiveParameter(f64),
    #[error("probability must be between 0.0 and 1.0; received {0}")]
    Probability(f64),
    /// Wraps the error returned by `rand_distr::Normal::new`.
    #[error(transparent)]
    Gaussian(#[from] rand_distr::NormalError),
}

/// Errors that can occur inside a crossover operator.
#[derive(Debug, Error)]
pub enum CrossoverError {
    /// Wraps the error returned by `ndarray::ShapeError`.
    #[error(transparent)]
    ShapeMismatch(#[from] ndarray::ShapeError),
}

/// Errors that can occur inside a sampling operator.
#[derive(Debug, Error)]
pub enum SamplingError {
    #[error(transparent)]
    ShapeMismatch(#[from] ndarray::ShapeError),
}

#[derive(Debug, Error)]
pub enum SurvivalError {
    #[error("value must be positive; received {0}")]
    PositiveParameter(f64),
    #[error("probability must be between 0.0 and 1.0; received {0}")]
    Probability(f64),
    /// When the score vector and the population have different lengths.
    #[error("score length ({score_len}) does not match population size ({population_len})")]
    LengthMismatch {
        score_len: usize,
        population_len: usize,
    },
    /// Wraps the error returned by `ndarray::ShapeError`.
    #[error(transparent)]
    ShapeMismatch(#[from] ndarray::ShapeError),
    /// Wraps the error returned by `ndarray_stats::errors::MinMaxError`.
    #[error(transparent)]
    MinMax(#[from] ndarray_stats::errors::MinMaxError),
}

/// Generic error type exposed by all genetic operators and the algorithms
/// that compose them.
///
/// Thanks to the `#[from]` attribute, any `MutationError`, `CrossoverError`,
/// or `SamplingError` can be automatically promoted to `OperatorError`
/// with the `?` operator.
#[derive(Debug, Error)]
pub enum OperatorError {
    #[error("No offspring were generated in the mating process")]
    EmptyMatingResult,
    #[error("mutation error: {0}")]
    Mutation(#[from] MutationError),
    #[error("crossover error: {0}")]
    Crossover(#[from] CrossoverError),
    #[error("sampling error: {0}")]
    Sampling(#[from] SamplingError),
    #[error("survival error: {0}")]
    Survival(#[from] SurvivalError),
}
