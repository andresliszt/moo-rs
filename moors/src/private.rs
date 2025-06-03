//! Module `private`
//!
//! This module defines “sealed” traits to restrict which array dimension types can be
//! used in various parts of the `moors` library. By sealing these traits, we ensure
//! that downstream users cannot implement them for arbitrary dimensions. Only the
//! intended `ndarray::Ix0`, `Ix1`, and `Ix2` types are allowed, preventing misuse
//! or unsupported dimension types in single‐ and multi‐objective contexts.

/// A sealed trait for 0‐ or 1‐dimensional arrays.
///
/// By implementing `SealedD01` only for `Ix0` and `Ix1`, we ensure that any generic
/// code bound by `T: SealedD01` can only accept `ndarray::Ix0` (scalar) or `ndarray::Ix1`
/// (vector). Users outside this crate cannot implement `SealedD01`, so no other
/// dimension can satisfy this bound.
pub trait SealedD01: ndarray::Dimension {}

impl SealedD01 for ndarray::Ix0 {}
impl SealedD01 for ndarray::Ix1 {}

/// A sealed trait for 1‐ or 2‐dimensional arrays with axis removal support.
///
/// By implementing `SealedD12` only for `Ix1` and `Ix2`, we ensure that any generic
/// code bound by `T: SealedD12` can only accept `ndarray::Ix1` (vector) or `ndarray::Ix2`
/// (matrix). The `RemoveAxis` supertrait guarantees that one can call `remove_axis`
/// on types that satisfy `SealedD12`. No downstream crate can add additional
/// implementations, so unsupported dimensions (e.g. `Ix3`) are excluded.
pub trait SealedD12: ndarray::Dimension + ndarray::RemoveAxis {}

impl SealedD12 for ndarray::Ix1 {}
impl SealedD12 for ndarray::Ix2 {}
