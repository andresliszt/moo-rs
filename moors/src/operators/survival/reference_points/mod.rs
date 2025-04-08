use ndarray::Array2;

mod dan_and_dennis;

pub use dan_and_dennis::DanAndDenisReferencePoints;

/// A common trait for structured reference points.
pub trait StructuredReferencePoints {
    fn generate(&self) -> Array2<f64>;
}
