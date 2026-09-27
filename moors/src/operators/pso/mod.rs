//! # PSO operators
//!
//! Generic trait definitions for particle-swarm algorithms, mirroring how
//! [`crate::operators`] provides pluggable building blocks for GAs. Concrete
//! implementations (e.g. those used by MO-ETPSO) live in their own module
//! instead of alongside the traits — mirroring how [`crate::algorithms::pso`]
//! splits the generic engine from each concrete algorithm.
//!
//! | Trait | Purpose | Typical implementations |
//! |-------|---------|--------------------------|
//! | [`VelocityUpdateOperator`] | Update every particle's velocity from `pbest`/`gbest`. | `ConstrictionVelocityUpdate` |
//! | [`MeritOperator`] | Score individuals (rank/crowding/constraints) to pick `pbest`/`gbest`. | `RankCrowdingMerit` |
pub mod merit;
pub(in crate::operators) mod mo_etpso;
pub mod velocity;

pub use merit::MeritOperator;
pub use mo_etpso::{ConstrictionVelocityUpdate, RankCrowdingMerit};
pub use velocity::VelocityUpdateOperator;
