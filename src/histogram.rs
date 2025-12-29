//! Traits for implementing histograms for Wang Landau or entropic sampling.
//! Contains histogram implementations for all primitive numbers
//! # Using Histograms
//! For the histograms of integers with bin width larger than 1: you should use the newly implemented
//! [GenericHist]  of [BinningWithWidth].
//! These will likely be much faster than [HistogramInt], especially if you have a lot of bins.
//! I did not remove the slower implementation, because then I'd have to change all
//! of my other code in which I use them ^^"
//!
//! Anyhow, here is an example for using a fast histogram with bin width 3
//!
//! ```
//! use sampling::histogram::*;
//! use rand_pcg::Pcg64;
//! use rand::prelude::*;
//! use rand::distr::*;
//!
//! // now I use one of the type aliases to first create the binning and then the histogram:
//! let mut hist = BinningI32::new_inclusive(-20,132, 3)
//!     .unwrap()
//!     .to_generic_hist();
//!
//! let uniform = Uniform::new_inclusive(-20, 132).unwrap();
//! let mut rng = Pcg64::seed_from_u64(3987612);
//! // create 10000 samples
//! let iter = uniform
//!     .sample_iter(rng)
//!     .take(10000);
//! for val in iter{
//!     hist.count_val(val)
//!         .unwrap(); // would panic if a value were to be outside the hist
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
//! use rand::distr::*;
//!
//! // now I use one of the type aliases to first create the binning and then the histogram:
//! let mut hist = FastBinningI32::new_inclusive(-20,130)
//!     .to_generic_hist();
//!
//! let uniform = Uniform::new_inclusive(-20, 130).unwrap();
//! let mut rng = Pcg64::seed_from_u64(3987612);
//! // create 10000 samples
//! let iter = uniform
//!     .sample_iter(rng)
//!     .take(10000);
//! for val in iter{
//!
//!     hist.count_val(val)
//!         .unwrap(); // would panic if a value were to be outside the hist
//!     // alternatively, if you don't want the panic:
//!     // let _ = hist.count_val(val);
//! }
//! ```
//!
//! # Atomic Histograms
//!
//! Sometimes you want to create a histograms in parallel, i.e.,
//! from multiple threads simultaneously.
//! In this case you can use Atomic histograms,
//! ```
//! use sampling::histogram::*;
//! use rand_pcg::Pcg64;
//! use rand::prelude::*;
//! use rand::distr::*;
//! use rayon::prelude::*;
//!
//! // now I use one of the type aliases to first create the binning and then the histogram:
//! let mut atomic_hist = BinningI16::new_inclusive(-20,132, 3)
//!     .unwrap()
//!     .to_generic_atomic_hist();
//!
//! let uniform = Uniform::new_inclusive(-20, 132)
//!     .unwrap();
//!
//! (0..4)
//!     .into_par_iter()
//!     .for_each(
//!         |seed|
//!         {
//!             let mut rng = Pcg64::seed_from_u64(seed);
//!             // create 10000 samples
//!             let iter = uniform
//!                 .sample_iter(rng)
//!                 .take(10000);
//!             for val in iter{
//!                 atomic_hist.count_val(val)
//!                     .unwrap(); // would panic if a value were to be outside the hist
//!                 // alternatively, if you don't want the panic:
//!                 // let _ = hist.count_val(val);
//!             }
//!         }
//!     );
//! assert_eq!(
//!     atomic_hist.total_hits(),
//!     40000
//! );
//!
//! // You can also convert the generic atomic histogram into a normal histogram.
//! let hist = atomic_hist.into_generic_hist();
//! // You can also convert it the other way round
//! let atomic_hist = hist.into_atomic_generic_hist();
//! ```

mod helper;
mod histogram_fast;
mod histogram_float;
mod histogram_int;
mod histogram_traits;

mod atomic_generic_hist;
mod atomic_hist_float;
mod binning;
mod generic_hist;

pub use helper::*;
pub use histogram_fast::*;
pub use histogram_float::*;
pub use histogram_int::*;
pub use histogram_traits::*;

pub use atomic_generic_hist::*;
pub use atomic_hist_float::*;
pub use binning::*;
pub use generic_hist::*;
