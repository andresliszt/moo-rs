use std::{error::Error, fmt};

use crate::evaluator::EvaluatorError;
use crate::operators::EvolveError;

/// Errors that can occur during initialization of the population.
#[derive(Debug)]
pub enum InitializationError {
    /// Error from the evaluator.
    Evaluator(EvaluatorError),
    /// Fitness array length does not match number of objectives.
    InvalidFitness(String),
    /// Constraints array length mismatch or unexpected absence.
    InvalidConstraints(String),
    /// Trying to access to algorithm attributes set after initialization
    NotInitializated(String),
}

impl fmt::Display for InitializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InitializationError::Evaluator(e) => {
                write!(f, "Error during evaluation in initialization: {}", e)
            }
            InitializationError::InvalidFitness(msg) => write!(f, "Invalid fitness setup: {}", msg),
            InitializationError::InvalidConstraints(msg) => {
                write!(f, "Invalid constraints setup: {}", msg)
            }
            InitializationError::NotInitializated(msg) => {
                write!(f, "Algorithm is not initialized yet: {}", msg)
            }
        }
    }
}

impl From<EvaluatorError> for InitializationError {
    fn from(e: EvaluatorError) -> Self {
        InitializationError::Evaluator(e)
    }
}

impl Error for InitializationError {}

#[derive(Debug)]
pub enum MultiObjectiveAlgorithmError {
    Evolve(EvolveError),
    Evaluator(EvaluatorError),
    InvalidParameter(String),
    Initialization(InitializationError),
}

impl fmt::Display for MultiObjectiveAlgorithmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultiObjectiveAlgorithmError::Evolve(msg) => {
                write!(f, "Error during evolution: {}", msg)
            }
            MultiObjectiveAlgorithmError::Evaluator(msg) => {
                write!(f, "Error during evaluation: {}", msg)
            }
            MultiObjectiveAlgorithmError::InvalidParameter(msg) => {
                write!(f, "Invalid parameter: {}", msg)
            }
            MultiObjectiveAlgorithmError::Initialization(msg) => {
                write!(f, "Error during initialization: {}", msg)
            }
        }
    }
}

impl From<EvolveError> for MultiObjectiveAlgorithmError {
    fn from(e: EvolveError) -> Self {
        MultiObjectiveAlgorithmError::Evolve(e)
    }
}

impl From<EvaluatorError> for MultiObjectiveAlgorithmError {
    fn from(e: EvaluatorError) -> Self {
        MultiObjectiveAlgorithmError::Evaluator(e)
    }
}

impl From<InitializationError> for MultiObjectiveAlgorithmError {
    fn from(e: InitializationError) -> Self {
        MultiObjectiveAlgorithmError::Initialization(e)
    }
}

impl Error for MultiObjectiveAlgorithmError {}
