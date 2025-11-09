//! # R‑NSGA‑II – Reference‑Point‑Guided NSGA‑II
//!
//! Implementation inspired by
//! **R. Imada, H. Ishibuchi & Y. Nojima,
//! “Reference Point–Based NSGA‑II for Preference‑Based Many‑Objective
//! Optimization”, in *Proc. GECCO 2017*, pp. 1923‑1930.**
//!
//! R‑NSGA‑II biases the classic NSGA‑II ranking/crowding procedure toward
//! **user‑supplied reference points**, enabling preference‑based search without
//! altering the core fast non‑dominated sorting.  At each generation it:
//!
//! 1. Performs NSGA‑II non‑dominated sorting.
//! 2. Computes a *reference‑distance* score (to the closest point in the user
//!    set) **to be *minimized***.
//! 3. Uses that score—rather than crowding distance—as the secondary key
//!    during survival selection.
//!
//! In *moors*, R‑NSGA‑II is wired from reusable operator bricks:
//!
//! * **Selection:** [`RankAndScoringSelection`] (`use_rank = true`, scoring **minimised**)
//! * **Survival:**  [`Rnsga2ReferencePointsSurvival`] (rank + reference‑distance)
//! * **Crossover / Mutation / Sampling:** user‑provided via the builder.
//!
//! Pass your reference point matrix (`Array2<f64>`) and an `epsilon` tolerance
//! to [`Rnsga2::new`]; the survivor will treat individuals whose distance to a
//! point ≤ ε as equally preferred.
//!
use ndarray::Array2;

use crate::{
    define_algorithm_and_builder,
    operators::{
        selection::moo::Rnsga2RankScoringSelection, survival::moo::Rnsga2ReferencePointsSurvival,
    },
};

define_algorithm_and_builder!(
    /// R-NSGA-II algorithm wrapper.
    ///
    /// Thin facade around [`GeneticAlgorithm`] pre-configured with
    /// reference-distance survival and a minimise-the-score selector.
    ///
    /// * **Selection:** [`RankAndScoringSelection`] (`SurvivalScoringComparison::Minimize`)
    /// * **Survival:**  [`Rnsga2ReferencePointsSurvival`]
    /// * **Paper:** Shinsaku Imada, Yuji Sakane, Masaru Tanigaki, and Masanori Sugimoto (2017),
    ///   "R-NSGA-II: Reference-Point-Based Non-Dominated Sorting Genetic Algorithm II",
    ///   *Proceedings of the Genetic and Evolutionary Computation Conference (GECCO ’17 Companion)*,
    ///   pp. 1685–1692, July 2017.
    ///   DOI: 10.1145/3067695.3082520
    ///
    /// Build via [`Rnsga2Builder`](crate::algorithms::Rnsga2Builder) or directly
    /// with [`Rnsga2::new`]; then call `run()` and `population()` to retrieve the
    /// preference-biased Pareto set.
    Rnsga2,
    Rnsga2RankScoringSelection,
    Rnsga2ReferencePointsSurvival,
    survival_args = [reference_points: Array2<f64>, epsilon: f64],
);
