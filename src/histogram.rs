//! Traits for implementing histograms for Wang Landau or entropic sampling.
//! Contains histogram implementations for all primitive numbers
//! # Note
//! For the histograms of integers with bin width larger than 1: you should use the newly implemented 
//! [GenericHist]  of [BinningWithWidth].
//! These will likely be much faster than [HistogramInt], especially if you have a lot of bins. 
//! I did not remove the slower implementation, because then I'd have to change all 
//! of my other code in which I use them ^^"
//! 
//! Anyhow, here is an example for using a fast histogram with bin width 1 
//! 
//! ```
//! use sampling::histogram::*;
//! use rand_pcg::Pcg64;
//! use rand::prelude::*;
//! use rand::distributions::*;
//! 
//! // now I use one of the type aliases to first create the binning and then the histogram:
//! let mut hist = BinningI32::new_inclusive(-20,132, 3)
//!     .unwrap()
//!     .to_generic_hist();
//! 
//! let uniform = Uniform::new_inclusive(-20, 132);
//! let mut rng = Pcg64::seed_from_u64(3987612);
//! // create 10000 samples
//! let iter = uniform
//!     .sample_iter(rng) 
//!     .take(10000);
//! for val in iter{
//!     hist.count_val(val)
//!         .unwrap(); // will panic if a value were to be outside the hist 
//!     // alternatively, if you don't want the panic:
//!     // let _ = hist.count_val(val);
//! }
//! ```
//! If you have a bin width of 1, then you can either use the newly implemented 
//! [GenericHist] of [FastSingleIntBinning] like in the example below, or 
//! you can keep using the old [HistogramFast], as there seems to be no real difference in speed,
//! at least on my machine they are within variance of one another.
//! 
//! Either way, there are type aliases for convenience, see below.
//! 
//! ```
//! use sampling::histogram::*;
//! use rand_pcg::Pcg64;
//! use rand::prelude::*;
//! use rand::distributions::*;
//! 
//! // now I use one of the type aliases to first create the binning and then the histogram:
//! let mut hist = FastBinningI32::new_inclusive(-20,130)
//!     .to_generic_hist();
//! 
//! let uniform = Uniform::new_inclusive(-20, 130);
//! let mut rng = Pcg64::seed_from_u64(3987612);
//! // create 10000 samples
//! let iter = uniform
//!     .sample_iter(rng) 
//!     .take(10000);
//! for val in iter{
//! 
//!     hist.count_val(val)
//!         .unwrap(); // will panic if a value were to be outside the hist 
//!     // alternatively, if you don't want the panic:
//!     // let _ = hist.count_val(val);
//! }
//! ```


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