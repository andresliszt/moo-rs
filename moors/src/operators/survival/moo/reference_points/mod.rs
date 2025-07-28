use ndarray::Array2;

pub mod dan_and_dennis;

pub use dan_and_dennis::*;

/// A common trait for structured reference points.
pub trait StructuredReferencePoints {
    fn generate(&self) -> Array2<f64>;
}
