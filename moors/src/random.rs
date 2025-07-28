//! # `random` – Unified RNG Abstraction
//!
//! All stochastic operators in *moors* (sampling, crossover, mutation,
//! selection, …) depend on an RNG, but you shouldn’t have to juggle the `rand`
//! crate’s API at every call site.  The **`RandomGenerator`** trait offers a
//! *single, ergonomic façade* while still giving power users access to the
//! underlying RNG when needed.
//!
//! ## Trait highlights
//!
//! | Method | Purpose |
//! |--------|---------|
//! | `gen_range_usize(min, max)` | `usize` in `[min, max)` |
//! | `gen_range_f64(min, max)`   | `f64`  in `[min, max)` |
//! | `gen_usize()`               | full‑range `usize` |
//! | `gen_bool(p)`               | Bernoulli(`p`) |
//! | `gen_probability()`          | uniform `[0, 1)` *(NB: typo kept for back‑compat)* |
//! | `shuffle_vec`, `shuffle_vec_usize` | in‑place Fisher–Yates |
//! | `choose_usize(slice)`       | random element or `None` |
//! | `rng()`                     | mutable handle to the raw `RngCore` object |
//!
//! The blanket implementations inside algorithms call these helpers—so you can
//! swap RNG engines, seed values, or mock objects *without modifying operator
//! code*.
//!
//! ## Ready‑to‑use generators
//!
//! | Type | Backed by | Intended for |
//! |------|-----------|--------------|
//! | [`MOORandomGenerator`] | `rand::rngs::StdRng` (ChaCha 12) | **Production**—fast, reproducible with a seed. |
//! | [`NoopRandomGenerator`] + `TestDummyRng` | stub → panics on direct RNG calls | **Unit tests** where randomness isn’t exercised but the trait is required. |
//!
//! ```rust
//! use moors::random::{MOORandomGenerator, RandomGenerator};
//! use rand::SeedableRng;
//!
//! // Deterministic RNG for reproducible experiments
//! let seed = 42_u64;
//! let mut rng = MOORandomGenerator::new_from_seed(Some(seed));
//!
//! // Flip a biased coin
//! if rng.gen_bool(0.1) {
//!     println!("Lucky mutation!");
//! }
//! ```
//!
//! ## Why not use `rand` directly everywhere?
//!
//! 1. **Centralised helpers** keep operator code concise and consistent.
//! 2. **Testing** – the `NoopRandomGenerator` lets you compile operator logic
//!    that never dereferences the RNG (e.g. deterministically seeded crossovers)
//!    while catching unintended randomness via a panic.
//!
//! ---
//!
use rand::prelude::IndexedRandom;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng};

/// A trait defining a unified interface for generating random values,
/// used across genetic operators and algorithms.
pub trait RandomGenerator {
    type R: RngCore + Rng;

    /// Generates a random `usize` in the range `[min, max)` using the underlying RNG.
    fn gen_range_usize(&mut self, min: usize, max: usize) -> usize {
        self.rng().random_range(min..max)
    }

    /// Generates a random `f64` in the range `[min, max)` using the underlying RNG.
    fn gen_range_f64(&mut self, min: f64, max: f64) -> f64 {
        self.rng().random_range(min..max)
    }

    /// Generates a random `usize` using the underlying RNG.
    fn gen_usize(&mut self) -> usize {
        self.rng().random_range(usize::MIN..usize::MAX)
    }

    /// Generates a random boolean value with probability `p` of being `true`
    /// using the underlying RNG.
    fn gen_bool(&mut self, p: f64) -> bool {
        self.rng().random_bool(p)
    }
    /// Generates a random probability as an `f64` in the range `[0.0, 1.0)`.
    fn gen_probability(&mut self) -> f64 {
        self.rng().random::<f64>()
    }
    fn shuffle_vec(&mut self, vector: &mut Vec<f64>) {
        vector.shuffle(self.rng())
    }
    fn shuffle_vec_usize(&mut self, vector: &mut Vec<usize>) {
        vector.shuffle(self.rng())
    }

    fn choose_usize<'a>(&mut self, vector: &'a [usize]) -> Option<&'a usize> {
        vector.choose(self.rng())
    }
    /// Returns a mutable reference to the underlying RNG implementing `RngCore`.
    fn rng(&mut self) -> &mut Self::R;
}

/// The production implementation of `RandomGenerator` using `StdRng`.
#[derive(Debug, Clone)]
pub struct MOORandomGenerator {
    rng: StdRng,
}

impl MOORandomGenerator {
    /// Creates a new `MOORandomGenerator` with the provided `StdRng`.
    pub fn new(rng: StdRng) -> Self {
        Self { rng }
    }
    pub fn new_from_seed(seed: Option<u64>) -> Self {
        let rng = seed.map_or_else(|| StdRng::from_rng(&mut rand::rng()), StdRng::seed_from_u64);
        Self { rng }
    }
}

impl RandomGenerator for MOORandomGenerator {
    type R = StdRng;
    /// Returns a mutable reference to the underlying `StdRng`.
    fn rng(&mut self) -> &mut StdRng {
        &mut self.rng
    }
}

/// A dummy implementation of `RandomGenerator` for testing purposes.
/// This struct is used when methods are called via the `RandomGenerator` trait
/// without directly interacting with self.rng. This is for testing only, see several
/// examples in the operators module
pub struct TestDummyRng;

impl RngCore for TestDummyRng {
    /// Not used in tests. This method is unimplemented.
    fn next_u32(&mut self) -> u32 {
        unimplemented!("Not used in this test")
    }

    /// Not used in tests. This method is unimplemented.
    fn next_u64(&mut self) -> u64 {
        unimplemented!("Not used in this test")
    }

    /// Not used in tests. This method is unimplemented.
    fn fill_bytes(&mut self, _dest: &mut [u8]) {
        unimplemented!("Not used in this test")
    }
}

pub struct NoopRandomGenerator {
    dummy: TestDummyRng,
}

impl Default for NoopRandomGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl NoopRandomGenerator {
    // @oliveira-sh: this is being used on tests
    pub fn new() -> Self {
        Self {
            dummy: TestDummyRng,
        }
    }
}

impl RandomGenerator for NoopRandomGenerator {
    type R = TestDummyRng;
    fn rng(&mut self) -> &mut TestDummyRng {
        &mut self.dummy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_gen_range_usize() {
        // Create a MOORandomGenerator with a fixed seed.
        let seed = [42u8; 32];
        let mut rng = MOORandomGenerator::new(StdRng::from_seed(seed));

        let min = 10;
        let max = 20;
        let value = rng.gen_range_usize(min, max);

        // Check that the generated value is within [min, max).
        assert!(
            value >= min && value < max,
            "gen_range_usize produced {} which is not in [{}, {})",
            value,
            min,
            max
        );
    }

    #[test]
    fn test_gen_range_f64() {
        let seed = [42u8; 32];
        let mut rng = MOORandomGenerator::new(StdRng::from_seed(seed));

        let min = 3.0;
        let max = 10.0;
        let value = rng.gen_range_f64(min, max);

        // Check that the generated value is within [min, max).
        assert!(
            value >= min && value < max,
            "gen_range_f64 produced {} which is not in [{}, {})",
            value,
            min,
            max
        );
    }

    #[test]
    fn test_gen_bool() {
        let seed = [42u8; 32];
        let mut rng = MOORandomGenerator::new(StdRng::from_seed(seed));

        // With a probability of 1.0, it should always return true.
        assert!(rng.gen_bool(1.0), "gen_bool(1.0) did not return true");

        // With a probability of 0.0, it should always return false.
        assert!(!rng.gen_bool(0.0), "gen_bool(0.0) did not return false");
    }

    #[test]
    fn test_gen_probability() {
        let seed = [42u8; 32];
        let mut rng = MOORandomGenerator::new(StdRng::from_seed(seed));

        // gen_probability (note the method name) returns a f64 in the range [0, 1).
        let prob = rng.gen_probability();
        assert!(
            prob >= 0.0 && prob < 1.0,
            "gen_probability produced {} which is not in [0, 1)",
            prob
        );
    }
}
