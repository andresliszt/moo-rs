use ndarray::{Array1, Array2, ArrayViewMut1, Axis, Ix1, Ix2, s};
use numpy::{IntoPyArray, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use moors::{
    AdaptiveController, AlgorithmContext, ControlSignal, CrossoverOperator, MutationOperator,
    Population, RandomGenerator, SamplingOperator,
};

fn select_individuals_idx(
    population_size: usize,
    rate: f64,
    rng: &mut impl RandomGenerator,
) -> Vec<usize> {
    let mask: Vec<bool> = (0..population_size).map(|_| rng.gen_bool(rate)).collect();
    let sel: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| if b { Some(i) } else { None })
        .collect();
    sel
}

/// Wrapper for a custom Python mutation operator.
///
/// This wrapper delegates population-level mutation to a Python-side class by
/// overriding the `operate` method. By operating on the entire population at
/// once, it acquires the GIL only once per call, improving performance compared
/// to invoking Python for each individual. The inner `PyObject` is expected to
/// be a Python class instance defining an `operate` method that takes a NumPy
/// array of shape (n_individuals, n_genes) and returns a mutated NumPy array of
/// the same shape.
#[derive(Debug)]
pub struct CustomPyMutationOperatorWrapper {
    pub inner: Py<PyAny>,
}

impl MutationOperator for CustomPyMutationOperatorWrapper {
    fn mutate<'a>(&self, mut _individual: ArrayViewMut1<'a, f64>, _rng: &mut impl RandomGenerator) {
        unimplemented!("Custom mutation operator overwrites operate method only")
    }

    fn operate(
        &self,
        population: &mut Array2<f64>,
        mutation_rate: f64,
        rng: &mut impl RandomGenerator,
    ) {
        // Acquire the GIL and convert our Rust view into a NumPy array...
        Python::attach(|py| {
            let population_size = population.nrows();
            let sel = select_individuals_idx(population_size, mutation_rate, rng);
            let filtered_population = population.select(Axis(0), &sel);
            let population_py = filtered_population.into_pyarray(py);

            // Call the Python-side operate method
            let mutated_population = self
                .inner
                .call_method1(py, "operate", (population_py,))
                .expect("Error calling custom mutation operate");

            let mutated_pyarray = mutated_population
                .bind(py)
                .cast::<PyArray2<f64>>()
                .expect("Expected a 2D float64 array, output of the operate method");

            let readonly: numpy::PyReadonlyArray2<'_, f64> = mutated_pyarray.readonly();
            let rust_view = readonly.as_array();
            for (mutated_row, &orig_idx) in rust_view.outer_iter().zip(&sel) {
                population.slice_mut(s![orig_idx, ..]).assign(&mutated_row);
            }
        });
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for CustomPyMutationOperatorWrapper {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, pyo3::PyAny>) -> Result<Self, Self::Error> {
        if !ob.hasattr("operate")? {
            return Err(PyTypeError::new_err(
                "Custom mutation operator class must define a 'operate' method",
            ));
        }
        Ok(CustomPyMutationOperatorWrapper {
            inner: ob.to_owned().unbind(),
        })
    }
}

/// Wrapper for a custom Python crossover operator.
///
/// Delegates population-level crossover to a Python-side class by overriding
/// the `operate` method. Only one GIL acquisition per mating process,
/// avoiding per-individual overhead. The inner `PyObject` is expected to be a
/// Python class instance defining `operate` that takes two NumPy arrays for
/// parents_a and parents_b and returns a NumPy array of offsprings.
#[derive(Debug)]
pub struct CustomPyCrossoverOperatorWrapper {
    pub inner: Py<PyAny>,
}

impl CrossoverOperator for CustomPyCrossoverOperatorWrapper {
    fn crossover(
        &self,
        _parent_a: &Array1<f64>,
        _parent_b: &Array1<f64>,
        _rng: &mut impl RandomGenerator,
    ) -> (Array1<f64>, Array1<f64>) {
        unimplemented!("Custom crossover operator overwrites operate method only")
    }

    fn operate(
        &self,
        parents_a: &Array2<f64>,
        parents_b: &Array2<f64>,
        cossover_rate: f64,
        rng: &mut impl RandomGenerator,
    ) -> Array2<f64> {
        Python::attach(|py| {
            let population_size = parents_a.nrows();
            // Build the mask with the mutation rate
            let sel = select_individuals_idx(population_size, cossover_rate, rng);
            let filtered_parents_a = parents_a.select(Axis(0), &sel);
            let filtered_parents_b = parents_b.select(Axis(0), &sel);
            let filtered_parents_a_py = filtered_parents_a.into_pyarray(py);
            let filtered_parents_b_py = filtered_parents_b.into_pyarray(py);
            // Call the Python-side operate method
            let offsprings = self
                .inner
                .call_method1(
                    py,
                    "operate",
                    (filtered_parents_a_py, filtered_parents_b_py),
                )
                .expect("Error calling custom crossover operate");

            let offsprings_pyarray = offsprings
                .bind(py)
                .cast::<PyArray2<f64>>()
                .expect("Expected a 2D float64 array, output of the operate method");

            let offsprings_rust = offsprings_pyarray.to_owned_array();
            return offsprings_rust;
        })
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for CustomPyCrossoverOperatorWrapper {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, pyo3::PyAny>) -> Result<Self, Self::Error> {
        if !ob.hasattr("operate")? {
            return Err(PyTypeError::new_err(
                "Custom mutation operator class must define a 'operate' method",
            ));
        }
        Ok(CustomPyCrossoverOperatorWrapper {
            inner: ob.to_owned().unbind(),
        })
    }
}

/// Wrapper for a custom Python sampling operator.
///
/// Delegates population-level sampling to a Python-side class by overriding
/// the `operate` method. Acquires the GIL once to invoke Python, expecting inner
/// Python class instance with `operate` method returning a NumPy array of
/// samples of shape (population_size, num_vars).
#[derive(Debug)]
pub struct CustomPySamplingOperatorWrapper {
    pub inner: Py<PyAny>,
}

impl SamplingOperator for CustomPySamplingOperatorWrapper {
    fn sample_individual(&self, _num_vars: usize, _rng: &mut impl RandomGenerator) -> Array1<f64> {
        unimplemented!("Custom sampling operator overwrites operate method only")
    }

    fn operate(
        &self,
        _population_size: usize,
        _num_vars: usize,
        _rng: &mut impl RandomGenerator,
    ) -> Array2<f64> {
        Python::attach(|py| {
            // Call the Python-side operate method
            let sample = self
                .inner
                .call_method0(py, "operate")
                .expect("Error calling custom sampling operate");

            let sample_pyarray = sample
                .bind(py)
                .cast::<PyArray2<f64>>()
                .expect("Expected a 2D float64 array, output of the operate method");

            let sample_rust = sample_pyarray.to_owned_array();
            return sample_rust;
        })
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for CustomPySamplingOperatorWrapper {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, pyo3::PyAny>) -> Result<Self, Self::Error> {
        if !ob.hasattr("operate")? {
            return Err(PyTypeError::new_err(
                "Custom sampling operator class must define an 'operate' method",
            ));
        }
        Ok(CustomPySamplingOperatorWrapper {
            inner: ob.to_owned().unbind(),
        })
    }
}

/// Wrapper for a custom Python adaptive controller.
///
/// Delegates per-generation observation to a Python-side class defining an
/// `observe` method with signature
/// `observe(iteration, genes, fitness, constraints, context) -> dict | None`,
/// where `context` is a dict with keys `num_vars`, `population_size`,
/// `num_offsprings`, `num_iterations`, `current_iteration`, `upper_bound`,
/// `lower_bound`. The returned dict may set `mutation_rate` (float),
/// `crossover_rate` (float) and/or `stop` (bool); omitted keys (or `None`)
/// leave the corresponding value unchanged, and `stop` defaults to `False`.
#[derive(Debug)]
pub struct CustomPyControllerWrapper {
    pub inner: Py<PyAny>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for CustomPyControllerWrapper {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, pyo3::PyAny>) -> Result<Self, Self::Error> {
        if !ob.hasattr("observe")? {
            return Err(PyTypeError::new_err(
                "Custom controller class must define an 'observe' method",
            ));
        }
        Ok(CustomPyControllerWrapper {
            inner: ob.to_owned().unbind(),
        })
    }
}

/// Converts an optional Python-side controller into a wrapper, wiring it into
/// the algorithm builder. Returns `None` (i.e. keep the default `NoController`)
/// when `py_obj_opt` is `None`.
pub fn controller_from_python(
    py_obj_opt: Option<Py<PyAny>>,
) -> PyResult<Option<CustomPyControllerWrapper>> {
    match py_obj_opt {
        None => Ok(None),
        Some(py_obj) => {
            Python::attach(|py| Ok(Some(py_obj.extract::<CustomPyControllerWrapper>(py)?)))
        }
    }
}

macro_rules! impl_custom_py_controller {
    ($fdim:ty) => {
        impl AdaptiveController<$fdim, Ix2> for CustomPyControllerWrapper {
            fn observe(
                &mut self,
                iteration: usize,
                population: &Population<$fdim, Ix2>,
                context: &AlgorithmContext,
            ) -> ControlSignal {
                Python::attach(|py| {
                    let genes_py = population.genes.to_pyarray(py);
                    let fitness_py = population.fitness.to_pyarray(py);
                    let constraints_py = population.constraints.to_pyarray(py);

                    let context_dict = PyDict::new(py);
                    context_dict
                        .set_item("num_vars", context.num_vars)
                        .expect("failed to build controller context dict");
                    context_dict
                        .set_item("population_size", context.population_size)
                        .expect("failed to build controller context dict");
                    context_dict
                        .set_item("num_offsprings", context.num_offsprings)
                        .expect("failed to build controller context dict");
                    context_dict
                        .set_item("num_iterations", context.num_iterations)
                        .expect("failed to build controller context dict");
                    context_dict
                        .set_item("current_iteration", context.current_iteration)
                        .expect("failed to build controller context dict");
                    context_dict
                        .set_item("upper_bound", context.upper_bound)
                        .expect("failed to build controller context dict");
                    context_dict
                        .set_item("lower_bound", context.lower_bound)
                        .expect("failed to build controller context dict");

                    let result = self
                        .inner
                        .call_method1(
                            py,
                            "observe",
                            (
                                iteration,
                                genes_py,
                                fitness_py,
                                constraints_py,
                                context_dict,
                            ),
                        )
                        .expect("Error calling custom controller observe");

                    if result.is_none(py) {
                        return ControlSignal::default();
                    }

                    let result_dict = result
                        .bind(py)
                        .cast::<PyDict>()
                        .expect("Expected a dict (or None), output of the observe method");

                    let mutation_rate = result_dict
                        .get_item("mutation_rate")
                        .expect("failed to read mutation_rate")
                        .filter(|v| !v.is_none())
                        .map(|v| v.extract::<f64>().expect("mutation_rate must be a float"));
                    let crossover_rate = result_dict
                        .get_item("crossover_rate")
                        .expect("failed to read crossover_rate")
                        .filter(|v| !v.is_none())
                        .map(|v| v.extract::<f64>().expect("crossover_rate must be a float"));
                    let stop = result_dict
                        .get_item("stop")
                        .expect("failed to read stop")
                        .filter(|v| !v.is_none())
                        .map(|v| v.extract::<bool>().expect("stop must be a bool"))
                        .unwrap_or(false);

                    ControlSignal {
                        mutation_rate,
                        crossover_rate,
                        stop,
                    }
                })
            }
        }
    };
}

impl_custom_py_controller!(Ix1);
impl_custom_py_controller!(Ix2);
