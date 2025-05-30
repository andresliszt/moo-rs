use moors::algorithms::MultiObjectiveAlgorithmError;
use moors::evaluator::EvaluatorError;
use pyo3::PyErr;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyRuntimeError};

// Raise this error when no feasible individuals are found
create_exception!(
    pymoors,
    NoFeasibleIndividualsError,
    PyException,
    "Raise this error when no feasible individuals are found"
);

// Raised when an invalid parameter value is provided
create_exception!(
    pymoors,
    InvalidParameterError,
    PyException,
    "Raised when an invalid parameter value is provided"
);

// Raised when an invalid parameter value is provided
create_exception!(
    pymoors,
    EmptyMatingError,
    PyException,
    "Raised when in some itearion no offsprings were generated"
);

create_exception!(
    pymoors,
    InitializationError,
    PyException,
    "Raised when accessing an algorithm that has not been initialized"
);

/// A local wrapper for MultiObjectiveAlgorithmError,
/// allowing us to implement conversion traits.
#[derive(Debug)]
pub struct MultiObjectiveAlgorithmErrorWrapper(pub MultiObjectiveAlgorithmError);

impl From<MultiObjectiveAlgorithmError> for MultiObjectiveAlgorithmErrorWrapper {
    fn from(err: MultiObjectiveAlgorithmError) -> Self {
        MultiObjectiveAlgorithmErrorWrapper(err)
    }
}

/// Once a new error is created to be exposed to the python side
/// the match must be updated to convert the error to the new error type.
impl From<MultiObjectiveAlgorithmErrorWrapper> for PyErr {
    fn from(err: MultiObjectiveAlgorithmErrorWrapper) -> PyErr {
        let msg = err.0.to_string();
        match err.0 {
            MultiObjectiveAlgorithmError::Initialization(_) => InitializationError::new_err(msg),
            MultiObjectiveAlgorithmError::Evaluator(EvaluatorError::NoFeasibleIndividuals) => {
                NoFeasibleIndividualsError::new_err(msg)
            }
            MultiObjectiveAlgorithmError::InvalidParameter(_) => {
                InvalidParameterError::new_err(msg)
            }
            _ => PyRuntimeError::new_err(msg),
        }
    }
}
