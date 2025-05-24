use thiserror::Error;

use crate::evaluator::EvaluatorError;
use crate::operators::evolve::EvolveError;

/// Errors that can occur during initialization of the population.
#[derive(Debug, Error)]
pub enum InitializationError {
    /// Error from the evaluator.
    #[error("Error during evaluation at initialization: {0}")]
    Evaluator(#[from] EvaluatorError),
    /// Fitness array length does not match number of objectives.
    #[error("Invalid fitness setup: {0}")]
    InvalidFitness(String),
    #[error("Invalid constraints setup: {0}")]
    InvalidConstraints(String),
    /// Constraints array length mismatch or unexpected absence.
    #[error("Algorithm is not initialized yet: {0}")]
    NotInitializated(String),
}

#[derive(Debug, Error)]
pub enum MultiObjectiveAlgorithmError {
    #[error("Error during evolution: {0}")]
    Evolve(#[from] EvolveError),
    #[error("Error during evaluation: {0}")]
    Evaluator(#[from] EvaluatorError),
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("Error during onitialization: {0}")]
    Initialization(#[from] InitializationError),
}
