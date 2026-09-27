//! Velocity update rules for particle-swarm algorithms.
use ndarray::Array2;

use crate::random::RandomGenerator;

/// Computes the new velocity of every particle from its current position and
/// velocity, its own personal best (`pbest`) and the swarm's global best
/// (`gbest`).
pub trait VelocityUpdateOperator: std::fmt::Debug {
    /// `gbest_position` is a `1 x num_vars` array (single row, broadcast
    /// against every particle).
    fn operate(
        &self,
        positions: &Array2<f64>,
        velocities: &Array2<f64>,
        pbest_positions: &Array2<f64>,
        gbest_position: &Array2<f64>,
        rng: &mut impl RandomGenerator,
    ) -> Array2<f64>;
}
