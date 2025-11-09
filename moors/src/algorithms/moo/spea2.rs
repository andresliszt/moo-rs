//! # SPEA‑2 – Strength‑Pareto Evolutionary Algorithm II
//!
//! Implementation of
//! **Eckart Zitzler, Marco Laumanns & Lothar Thiele,
//! “SPEA2: Improving the Strength Pareto Evolutionary Algorithm”,
//! Technical Report 103, Computer Engineering and Networks Laboratory (TIK),
//! ETH Zürich, 2001.**
//!
//! SPEA‑2 maintains two populations (archive + current) and assigns each
//! individual a raw‑fitness value derived from *how many* solutions it dominates
//! (*strength*) and *by how many* it is dominated.  Diversity is preserved with
//! a *k‑nearest‑neighbour* density estimator.
//!
//! In *moors*, SPEA‑2 is wired from reusable operator bricks:
//!
//! * **Selection:** [`RankAndScoringSelection`] (only survival‑score is used)
//! * **Survival:**  [`Spea2KnnSurvival`] (strength + k‑NN density)
//! * **Crossover / Mutation / Sampling:** user‑provided via the builder.
//!
//! The default configuration keeps a secondary **archive** whose size equals
//! the main population; truncation is handled by the k‑NN density measure.
//!

use crate::{
    define_algorithm_and_builder,
    operators::{selection::moo::Spea2ScoringSelection, survival::moo::Spea2KnnSurvival},
};

define_algorithm_and_builder!(
    /// SPEA-II algorithm wrapper.
    ///
    /// This struct is a thin facade over [`GeneticAlgorithm`] preset with
    /// the SPEA-II survival and selection strategy.
    ///
    /// * **Selection:** [`RankAndScoringSelection`]
    /// * **Survival:**  [`Spea2KnnSurvival`] (elitist, k-nearest neighbors density)
    ///
    /// Construct it with [`Spea2Builder`](crate::algorithms::Spea2Builder).
    /// After building, call [`run`](GeneticAlgorithm::run)
    /// and then [`population`](GeneticAlgorithm::population) to retrieve the
    /// final non-dominated set.
    ///
    /// For algorithmic details, see:
    /// Eckart Zitzler, Marco Laumanns, and Lothar Thiele (2001),
    /// "SPEA2: Improving the Strength Pareto Evolutionary Algorithm",
    /// TIK-Report 103, Computer Engineering and Networks Laboratory,
    /// ETH Zurich, Switzerland, 2001.
    Spea2,
    Spea2ScoringSelection,
    Spea2KnnSurvival
);
