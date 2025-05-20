use ndarray::{Axis, s};
use numpy::{IntoPyArray, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use moors::{
    genetic::{IndividualGenes, IndividualGenesMut, PopulationGenes},
    operators::{CrossoverOperator, GeneticOperator, MutationOperator},
    random::RandomGenerator,
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

#[derive(Debug)]
pub struct CustomPyMutationOperatorWrapper {
    pub inner: PyObject,
}

impl GeneticOperator for CustomPyMutationOperatorWrapper {
    fn name(&self) -> String {
        "CustomPyMutationOperatorWrapper".into()
    }
}

impl MutationOperator for CustomPyMutationOperatorWrapper {
    fn mutate<'a>(&self, mut _individual: IndividualGenesMut<'a>, _rng: &mut impl RandomGenerator) {
        unimplemented!("Custom mutation operator overwrites operate method only")
    }

    fn operate(
        &self,
        population: &mut PopulationGenes,
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
                .downcast::<PyArray2<f64>>() // ➜ Bound<'py, PyArray2<f64>>
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

#[derive(Debug)]
pub struct CustomPyCrossoverOperatorWrapper {
    pub inner: PyObject,
}

impl GeneticOperator for CustomPyCrossoverOperatorWrapper {
    fn name(&self) -> String {
        "CustomPyCrossoverOperatorWrapper".into()
    }
}

impl CrossoverOperator for CustomPyCrossoverOperatorWrapper {
    fn crossover(
        &self,
        _parent_a: &IndividualGenes,
        _parent_b: &IndividualGenes,
        _rng: &mut impl RandomGenerator,
    ) -> (
        moors::genetic::IndividualGenes,
        moors::genetic::IndividualGenes,
    ) {
        unimplemented!("Custom crossover operator overwrites operate method only")
    }

    fn operate(
        &self,
        parents_a: &PopulationGenes,
        parents_b: &PopulationGenes,
        cossover_rate: f64,
        rng: &mut impl RandomGenerator,
    ) -> PopulationGenes {
        // Acquire the GIL and convert our Rust view into a NumPy array...
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
                .downcast::<PyArray2<f64>>() // ➜ Bound<'py, PyArray2<f64>>
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
