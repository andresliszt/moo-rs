//! # MO-ETPSO – Multi-Objective Elitist-Tournament Particle Swarm Optimization
//!
//! A concrete, ready-to-use PSO variant built from reusable operator bricks:
//!
//! * **Velocity update:** [`ConstrictionVelocityUpdate`] — classical
//!   constriction-factor PSO (Clerc & Kennedy, 2002).
//! * **Merit:** [`RankCrowdingMerit`] — Pareto rank + product-based crowding
//!   distance minus constraint violation, used to pick each particle's
//!   `pbest` and the swarm's `gbest`.
//!
//! The public API exposes only the algorithm's own tunable parameters
//! (`inertia_weight`, `cognitive_coefficient`, `social_coefficient`); the
//! underlying operator structs are never constructed by the caller — mirroring
//! how, e.g., `Nsga2Builder` never asks the caller to build a survival
//! operator directly.
use crate::{
    define_pso_algorithm_and_builder,
    operators::{ConstrictionVelocityUpdate, RankCrowdingMerit},
};

define_pso_algorithm_and_builder!(
    /// MO-ETPSO algorithm wrapper.
    ///
    /// This type alias is a thin facade over [`Pso`](crate::algorithms::Pso)
    /// preset with the constriction-factor velocity update and the
    /// rank+crowding merit strategy.
    ///
    /// Construct it with [`MoEtpsoBuilder`]. After building, call
    /// [`run`](crate::algorithms::Pso::run) and then
    /// [`best_population`](crate::algorithms::Pso::best_population) to
    /// retrieve the swarm's elitist archive (its non-dominated set).
    MoEtpso, ConstrictionVelocityUpdate, RankCrowdingMerit,
    velocity_args = [
        /// Inertia weight `w` of the constriction-factor velocity update.
        /// Defaults to `1.0`.
        inertia_weight: f64 = 1.0,
        /// Cognitive coefficient `c1` (pull towards a particle's own best).
        /// Defaults to `2.05`.
        cognitive_coefficient: f64 = 2.05,
        /// Social coefficient `c2` (pull towards the swarm's best). Defaults
        /// to `2.05`.
        social_coefficient: f64 = 2.05,
    ],
);
