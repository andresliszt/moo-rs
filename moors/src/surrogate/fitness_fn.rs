//! Wraps a real [`FitnessFn`] with a [`SurrogateModel`], so that surrogate-assisted
//! evaluation is just "a fitness function" from the algorithm's point of view.
use std::sync::Mutex;

use ndarray::{Array1, Array2, ArrayBase, Axis, Ix1, Ix2, OwnedRepr};

use crate::evaluator::FitnessFn;
use crate::genetic::{D12, Fitness};

use super::SurrogateModel;

/// Converts a surrogate prediction matrix (`N x num_objectives`) back into the
/// wrapped function's native fitness representation (`Array1<f64>` for
/// single-objective, `Array2<f64>` for multi-objective). `D12` is sealed to
/// exactly these two dimensionalities, so this covers every valid `FitnessFn::Dim`.
pub(crate) trait MatrixConvert: D12 {
    fn from_matrix(matrix: Array2<f64>) -> Fitness<Self>;
}

impl MatrixConvert for Ix1 {
    fn from_matrix(matrix: Array2<f64>) -> Fitness<Ix1> {
        Array1::from_shape_vec(matrix.nrows(), matrix.iter().cloned().collect()).expect(
            "surrogate fitness matrix must have exactly one column for single-objective problems",
        )
    }
}

impl MatrixConvert for Ix2 {
    fn from_matrix(matrix: Array2<f64>) -> Fitness<Ix2> {
        matrix
    }
}

fn fitness_to_matrix<D: D12>(arr: &Fitness<D>) -> Array2<f64> {
    let n = arr.shape()[0];
    let cols = arr.shape().get(1).copied().unwrap_or(1);
    Array2::from_shape_vec((n, cols), arr.iter().cloned().collect())
        .expect("fitness array must have a valid shape")
}

/// Configuration for [`SurrogateFitnessFn`].
#[derive(Debug, Clone)]
pub struct SurrogateConfig {
    /// Fraction of each batch of genes evaluated with the real, potentially
    /// expensive fitness function; the rest gets the surrogate's prediction.
    pub infill_ratio: f64,
    /// Number of initial calls that always evaluate everyone for real, so the
    /// surrogate has a minimal archive to train on before it starts predicting.
    pub warmup_iterations: usize,
}

impl Default for SurrogateConfig {
    fn default() -> Self {
        Self {
            infill_ratio: 0.3,
            warmup_iterations: 5,
        }
    }
}

struct SurrogateState {
    model: Box<dyn SurrogateModel>,
    call_count: usize,
}

/// A [`FitnessFn`] that transparently approximates most evaluations with a
/// [`SurrogateModel`] after a warmup period, falling back to the real
/// (potentially expensive) function otherwise. Because it implements
/// `FitnessFn` itself, it can be passed directly to `.fitness_fn(...)` on any
/// algorithm builder — the genetic algorithm requires no changes at all.
///
/// Interior mutability (`Mutex`) is required because `FitnessFn::call` takes
/// `&self`; the mutex is only held for the duration of one generation's batch
/// evaluation, so contention is a non-issue.
pub struct SurrogateFitnessFn<F: FitnessFn> {
    real_fitness: F,
    state: Mutex<SurrogateState>,
    config: SurrogateConfig,
}

impl<F: FitnessFn> SurrogateFitnessFn<F> {
    pub fn new(
        real_fitness: F,
        model: impl SurrogateModel + 'static,
        config: SurrogateConfig,
    ) -> Self {
        Self {
            real_fitness,
            state: Mutex::new(SurrogateState {
                model: Box::new(model),
                call_count: 0,
            }),
            config,
        }
    }
}

impl<F: FitnessFn> FitnessFn for SurrogateFitnessFn<F>
where
    F::Dim: MatrixConvert,
{
    type Dim = F::Dim;

    fn call(&self, genes: &Array2<f64>) -> ArrayBase<OwnedRepr<f64>, Self::Dim> {
        let mut state = self.state.lock().expect("surrogate state mutex poisoned");
        state.call_count += 1;

        if state.call_count <= self.config.warmup_iterations || !state.model.is_fitted() {
            // Warmup, or the model hasn't actually retrained yet (e.g.
            // `retrain_every` > `warmup_iterations`): evaluate everyone for
            // real and feed the model rather than predicting from nothing.
            let real = self.real_fitness.call(genes);
            state.model.update(genes, &fitness_to_matrix(&real));
            return real;
        }

        let mut predicted = state.model.predict(genes);
        let n = genes.nrows();
        let k = (((n as f64) * self.config.infill_ratio).round() as usize).clamp(1, n);

        let infill_indices: Vec<usize> = match state.model.predict_std(genes) {
            // Prioritize the points the model is least sure about (Expected-Improvement-lite).
            Some(std) => {
                let mut scored: Vec<(usize, f64)> = (0..n).map(|i| (i, std.row(i).sum())).collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                scored.into_iter().take(k).map(|(i, _)| i).collect()
            }
            // No uncertainty estimate available: spread real evaluations evenly.
            None => {
                let stride = ((n as f64) / (k as f64)).floor().max(1.0) as usize;
                (0..n).step_by(stride).take(k).collect()
            }
        };

        let infill_genes = genes.select(Axis(0), &infill_indices);
        let real_fitness_subset = self.real_fitness.call(&infill_genes);
        let real_matrix = fitness_to_matrix(&real_fitness_subset);
        state.model.update(&infill_genes, &real_matrix);

        for (row, &idx) in infill_indices.iter().enumerate() {
            predicted.row_mut(idx).assign(&real_matrix.row(row));
        }

        F::Dim::from_matrix(predicted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, array};
    use std::cell::Cell;
    use std::rc::Rc;

    struct CountingModel {
        updates: Rc<Cell<usize>>,
    }

    impl SurrogateModel for CountingModel {
        fn update(&mut self, _genes: &Array2<f64>, _fitness: &Array2<f64>) {
            self.updates.set(self.updates.get() + 1);
        }
        fn predict(&self, genes: &Array2<f64>) -> Array2<f64> {
            Array2::zeros((genes.nrows(), 1))
        }
    }

    fn sphere(genes: &Array2<f64>) -> Array1<f64> {
        genes.map_axis(Axis(1), |row| row.iter().map(|&x| x * x).sum())
    }

    #[test]
    fn warmup_always_uses_real_function() {
        let updates = Rc::new(Cell::new(0));
        let surrogate = SurrogateFitnessFn::new(
            sphere,
            CountingModel {
                updates: updates.clone(),
            },
            SurrogateConfig {
                infill_ratio: 0.5,
                warmup_iterations: 2,
            },
        );

        let genes = array![[1.0], [2.0], [3.0]];
        let f1 = surrogate.call(&genes);
        let f2 = surrogate.call(&genes);

        assert_eq!(f1, array![1.0, 4.0, 9.0]);
        assert_eq!(f2, array![1.0, 4.0, 9.0]);
        assert_eq!(updates.get(), 2);
    }

    #[test]
    fn after_warmup_only_infill_subset_is_updated() {
        let updates = Rc::new(Cell::new(0));
        let surrogate = SurrogateFitnessFn::new(
            sphere,
            CountingModel {
                updates: updates.clone(),
            },
            SurrogateConfig {
                infill_ratio: 0.5,
                warmup_iterations: 0,
            },
        );

        let genes = array![[1.0], [2.0], [3.0], [4.0]];
        let result = surrogate.call(&genes);

        assert_eq!(result.len(), 4);
        assert_eq!(updates.get(), 1, "one update call for the infill subset");
    }
}
