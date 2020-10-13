//! # For sampling ensembles
//! * contains Simple sampling, WangLandau, entropic sampling, Metropolis, Histograms

/// Contains traits useful for sampling an ensemble
/// like MarkovChain or Metropolis etc.
pub mod traits;
pub mod metropolis_helper;
pub mod wang_landau;
pub mod histogram;
pub mod heatmap;
pub mod entropic_sampling;
pub mod glue;
#[cfg(feature="bootstrap")]
pub mod bootstrap;


pub use wang_landau::*;
pub use entropic_sampling::*;

pub use metropolis_helper::*;
pub use histogram::*;
pub use heatmap::*;
pub use glue::*;
#[cfg(feature="bootstrap")]
pub use bootstrap::*;