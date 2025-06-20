//! # REVEA – Reference‑Vector‑Guided Evolutionary Algorithm
//!
//! Implementation of
//! **Ran Cheng, Yaochu Jin, Markus Olhofer & Bernhard Sendhoff,
//! “A Reference Vector Guided Evolutionary Algorithm for Many‑Objective
//! Optimization”, IEEE Transactions on Evolutionary Computation 20 (5):
//! 773‑791 (2016).**
//!
//! REVEA tackles many‑objective problems by steering the population with a
//! **dynamic set of reference vectors**.  Each generation it:
//!
//! 1. Performs non‑dominated sorting (like NSGA‑II/III).
//! 2. Associates every solution to its nearest reference vector (angle‑based).
//! 3. Uses a **shift‑based density estimator** to pick survivors, balancing
//!    convergence and diversity.
//! 4. Periodically *rotates* or *re‑scales* the reference vectors
//!    (`frequency`) so the search can adapt to the true Pareto front shape.
//!
//! In *moors*, REVEA is wired from reusable operator bricks:
//!
//! * **Selection:** [`RandomSelection`] (uniform binary tournament)
//! * **Survival:**  [`ReveaReferencePointsSurvival`]
//! * **Crossover / Mutation / Sampling:** user‑provided via the builder.
//!
//! You pass an initial `Array2<f64>` of reference vectors plus two hyper‑
//! parameters to [`Revea::new`]:
//!
//! * `alpha`     – controls the rotation angle when updating vectors.
//! * `frequency` – how often (in generations) the reference set is refreshed.
//!

use crate::operators::{
    selection::moo::RandomSelection, survival::moo::ReveaReferencePointsSurvival,
};

create_algorithm!(
    /// REVEA algorithm wrapper.
    ///
    /// Thin facade around [`GeneticAlgorithmMOO`] pre-configured with
    /// reference-vector survival and random parent selection.
    ///
    /// * **Selection:** [`RandomSelection`]
    /// * **Survival:**  [`ReveaReferencePointsSurvival`]
    /// * **Paper:** Ran Cheng, Yixin Zhang, Min Dai, and Xingyi Zhang (2016),
    ///   "A Reference-Vector Guided Evolutionary Algorithm for Many-Objective Optimization",
    ///   *IEEE Transactions on Evolutionary Computation*, vol. 20, no. 5,
    ///   pp. 773–791, Oct. 2016.
    ///   DOI: 10.1109/TEVC.2015.2495854
    ///
    /// Build via [`ReveaBuilder`](crate::algorithms::ReveaBuilder) or directly with
    /// [`Revea::new`], then call `run()` and `population()` to obtain the final
    /// Pareto approximation.
    Revea,
    RandomSelection,
    ReveaReferencePointsSurvival
);

impl<S, Cross, Mut, F, G, DC> Default for ReveaBuilder<S, Cross, Mut, F, G, DC>
where
    S: SamplingOperator,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: FitnessFn<Dim = ndarray::Ix2>,
    G: ConstraintsFn,
    DC: PopulationCleaner,
    AlgorithmMOOBuilder<S, RandomSelection, ReveaReferencePointsSurvival, Cross, Mut, F, G, DC>:
        Default,
{
    fn default() -> Self {
        let mut inner: AlgorithmMOOBuilder<
            S,
            RandomSelection,
            ReveaReferencePointsSurvival,
            Cross,
            Mut,
            F,
            G,
            DC,
        > = Default::default();
        inner = inner.selector(RandomSelection);
        ReveaBuilder {
            inner_builder: inner,
        }
    }
}
