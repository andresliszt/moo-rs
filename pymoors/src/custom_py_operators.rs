use ndarray::{Array1, Array2, ArrayViewMut1, Axis, s};
use numpy::{IntoPyArray, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use moors::{CrossoverOperator, MutationOperator, RandomGenerator, SamplingOperator};

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
    pub inner: PyObject,
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
        Python::with_gil(|py| {
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
                .downcast::<PyArray2<f64>>()
                .expect("Expected a 2D float64 array, output of the operate method");

            let readonly: numpy::PyReadonlyArray2<'_, f64> = mutated_pyarray.readonly();
            let rust_view = readonly.as_array();
            for (mutated_row, &orig_idx) in rust_view.outer_iter().zip(&sel) {
                population.slice_mut(s![orig_idx, ..]).assign(&mutated_row);
            }
        });
    }
}

impl<'py> FromPyObject<'py> for CustomPyMutationOperatorWrapper {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if !ob.hasattr("operate")? {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Custom mutation operator class must define a 'operate' method",
            ));
        }
        Ok(CustomPyMutationOperatorWrapper {
            inner: ob.clone().unbind(),
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
    pub inner: PyObject,
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
        Python::with_gil(|py| {
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
                .downcast::<PyArray2<f64>>()
                .expect("Expected a 2D float64 array, output of the operate method");

            let offsprings_rust = offsprings_pyarray.to_owned_array();
            return offsprings_rust;
        })
    }
}

impl<'py> FromPyObject<'py> for CustomPyCrossoverOperatorWrapper {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if !ob.hasattr("operate")? {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Custom mutation operator class must define a 'operate' method",
            ));
        }
        Ok(CustomPyCrossoverOperatorWrapper {
            inner: ob.clone().unbind(),
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
    pub inner: PyObject,
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
        Python::with_gil(|py| {
            // Call the Python-side operate method
            let sample = self
                .inner
                .call_method0(py, "operate")
                .expect("Error calling custom sampling operate");

            let sample_pyarray = sample
                .bind(py)
                .downcast::<PyArray2<f64>>()
                .expect("Expected a 2D float64 array, output of the operate method");

            let sample_rust = sample_pyarray.to_owned_array();
            return sample_rust;
        })
    }
}

impl<'py> FromPyObject<'py> for CustomPySamplingOperatorWrapper {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if !ob.hasattr("operate")? {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Custom sampling operator class must define an 'operate' method",
            ));
        }
        Ok(CustomPySamplingOperatorWrapper {
            inner: ob.clone().unbind(),
        })
    }
}
