use std::{error::Error, fmt};

use crate::{
    algorithms::helpers::initialization::InitializationError, evaluator::EvaluatorError,
    operators::EvolveError,
};

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
