use moors::EvaluatorError;
use moors::{AlgorithmBuilderError, AlgorithmError};
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
pub struct AlgorithmErrorWrapper(pub AlgorithmError);

impl From<AlgorithmError> for AlgorithmErrorWrapper {
    fn from(err: AlgorithmError) -> Self {
        AlgorithmErrorWrapper(err)
    }
}

/// Once a new error is created to be exposed to the python side
/// the match must be updated to convert the error to the new error type.
impl From<AlgorithmErrorWrapper> for PyErr {
    fn from(err: AlgorithmErrorWrapper) -> PyErr {
        let msg = err.0.to_string();
        match err.0 {
            AlgorithmError::Initialization(_) => InitializationError::new_err(msg),
            AlgorithmError::Evaluator(EvaluatorError::NoFeasibleIndividuals) => {
                NoFeasibleIndividualsError::new_err(msg)
            }
            AlgorithmError::ValidationError(_) => InvalidParameterError::new_err(msg),
            _ => PyRuntimeError::new_err(msg),
        }
    }
}

impl From<AlgorithmBuilderError> for AlgorithmErrorWrapper {
    fn from(err: AlgorithmBuilderError) -> Self {
        // first into AlgorithmError, then wrap
        AlgorithmErrorWrapper(err.into())
    }
}
