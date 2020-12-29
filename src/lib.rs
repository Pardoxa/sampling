//! # For sampling ensembles
//! * contains Simple sampling, WangLandau, entropic sampling, Metropolis, Histograms
//! 
//! * [Detailed example](examples/coin_flips/index.html) for Wang-landau with sucessive entropic sampling
//! and comparison with analytical results

//#![deny(missing_docs, warnings)]

/// Contains traits useful for sampling an ensemble
/// like MarkovChain or Metropolis etc.
pub mod traits;

pub mod wang_landau;
pub mod histogram;
pub mod heatmap;
pub mod entropic_sampling;
pub mod glue;
#[cfg(feature="bootstrap")]
pub mod bootstrap;

#[cfg(feature="rewl")]
pub mod rewl;

pub mod metropolis;

pub use metropolis::*;

pub use wang_landau::*;
pub use entropic_sampling::*;


pub use histogram::*;
pub use heatmap::*;
pub use glue::*;
pub use traits::*;
#[cfg(feature="bootstrap")]
pub use bootstrap::*;

pub mod examples;