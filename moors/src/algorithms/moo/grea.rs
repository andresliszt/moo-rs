//! # GrEA – Grid-based Evolutionary Algorithm for Many-Objective Optimization
//!
//! Implementation of
//! **S. Yang, M. Li, X. Liu & J. Zheng,
//! "A Grid-Based Evolutionary Algorithm for Many-Objective Optimization",**
//! *IEEE Transactions on Evolutionary Computation*, 17(5):721-736, 2013.
//! DOI: 10.1109/TEVC.2012.2227145
//!
//! GrEA partitions objective space into a hyper-grid (`div` divisions per
//! objective) and uses three grid-based metrics — grid ranking (`GR`), grid
//! crowding distance (`GCD`) and grid coordinate point distance (`GCPD`) —
//! to guide both environmental (survival) and mating (tournament) selection
//! toward a well-spread, well-converged Pareto front, which tends to scale
//! better than crowding-distance-based approaches (e.g. NSGA-II) as the
//! number of objectives grows.
//!
//! In *moors*, GrEA is wired from reusable operator bricks:
//!
//! * **Selection:** [`GreaTournamentSelection`]
//! * **Survival:**  [`GreaSurvival`]
//! * **Crossover / Mutation / Sampling:** user-provided via the builder.
//!
//! Both operators need the same `div` (grid divisions) hyper-parameter,
//! wired via `shared_args` so [`GreaBuilder`] exposes a single `.div(usize)`
//! setter that feeds both [`GreaTournamentSelection::new`] and
//! [`GreaSurvival::new`].
use crate::{
    define_algorithm_and_builder,
    operators::{selection::moo::GreaTournamentSelection, survival::moo::GreaSurvival},
};

define_algorithm_and_builder!(
    /// GrEA algorithm wrapper.
    ///
    /// Thin facade around [`GeneticAlgorithm`] pre-configured with grid-based
    /// survival and mating selection.
    ///
    /// * **Selection:** [`GreaTournamentSelection`]
    /// * **Survival:**  [`GreaSurvival`]
    /// * **Paper:** Shengxiang Yang, Miqing Li, Xiaohui Liu, and Jinhua Zheng (2013),
    ///   "A Grid-Based Evolutionary Algorithm for Many-Objective Optimization",
    ///   *IEEE Transactions on Evolutionary Computation*, vol. 17, no. 5,
    ///   pp. 721-736, Oct. 2013.
    ///   DOI: 10.1109/TEVC.2012.2227145
    ///
    /// Build via [`GreaBuilder`], then call `run()` and `population()` to
    /// obtain the final Pareto approximation.
    Grea,
    GreaTournamentSelection,
    GreaSurvival,
    shared_args = [ div: usize ]
);
