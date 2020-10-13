//! Traits for implementing histograms for Wang Landau or entropic sampling.
//! Contains histogram implementations for all primitive numbers
mod histogram_traits;
mod histogram_float;
mod histogram_int;
mod helper;
mod histogram_fast;

pub use histogram_traits::*;
pub use histogram_float::*;
pub use histogram_int::*;
pub use helper::*;
pub use histogram_fast::*;