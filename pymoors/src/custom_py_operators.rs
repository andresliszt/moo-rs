use ndarray::Axis;
use numpy::{IntoPyArray, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use moors::{
    genetic::{IndividualGenesMut, PopulationGenes},
    operators::{GeneticOperator, MutationOperator},
    random::RandomGenerator,
};

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
            // Build the mask with the mutation rate
            let mask: Vec<bool> =
                self.select_individuals_for_mutation(population_size, mutation_rate, rng);
            // TODO: This sel additional step will be removed once this is done:
            //
            let sel: Vec<usize> = mask
                .iter()
                .enumerate()
                .filter_map(|(i, &b)| if b { Some(i) } else { None })
                .collect();
            let filtered_population = population.select(Axis(0), &sel);
            let population_py = filtered_population.to_owned().into_pyarray(py);

            // Call the Python-side mutate method
            let mutated_population = self
                .inner
                .call_method1(py, "operate", (population_py,))
                .expect("Error calling custom mutate");

            let mutated_pyarray = mutated_population
                .bind(py)
                .downcast::<PyArray2<f64>>() // ➜ Bound<'py, PyArray2<f64>>
                .expect("Expected a 2D float64 array, output of the mutate method");

            // get a readonly view of the PyArray as an `ndarray::ArrayVie21<f64>`
            let readonly: numpy::PyReadonlyArray2<'_, f64> = mutated_pyarray.readonly();
            let rust_view = readonly.as_array();
            println!("RUST VIEW {}", rust_view);
            println!("POPULATION {}", population);
            population.assign(&rust_view);
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
