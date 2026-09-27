//! # Particle Swarm variants
//!
//! This module gathers concrete, ready-to-use PSO algorithms built on top of
//! the generic engine [`crate::algorithms::Pso`] — the PSO counterpart of
//! [`crate::algorithms::moo`] for the GA family. Each concrete algorithm here
//! presets its own velocity-update and merit strategy and exposes only its
//! own tunable parameters, instead of requiring users to construct and pass
//! operator structs directly.
//!
//! | Algorithm | Velocity update | Merit | Builder type |
//! |-----------|------------------|-------|--------------|
//! | **MO-ETPSO** | [`ConstrictionVelocityUpdate`](crate::operators::ConstrictionVelocityUpdate) | [`RankCrowdingMerit`](crate::operators::RankCrowdingMerit) | [`MoEtpsoBuilder`](crate::algorithms::MoEtpsoBuilder) |

pub(in crate::algorithms) mod mo_etpso;
