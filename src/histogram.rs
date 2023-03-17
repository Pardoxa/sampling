//! Traits for implementing histograms for Wang Landau or entropic sampling.
//! Contains histogram implementations for all primitive numbers
mod histogram_traits;
mod histogram_float;
mod histogram_int;
mod helper;
mod histogram_fast;
mod atomic_hist_int;
mod atomic_hist_float;
mod binning;
mod generic_hist;
mod generic_atomic_hist;

pub use histogram_traits::*;
pub use histogram_float::*;
pub use histogram_int::*;
pub use helper::*;
pub use histogram_fast::*;
pub use atomic_hist_int::*;
pub use atomic_hist_float::*;
pub use binning::*;
pub use generic_hist::*;
pub use generic_atomic_hist::*;