//! # For sampling ensembles
//! * contains Simple sampling, WangLandau, entropic sampling, Metropolis, Histograms
//! 
//! * [Detailed example](examples/coin_flips/index.html) for Wang-landau with successive entropic sampling
//! and comparison with analytical results
//! * [Detailed example](examples/coin_flips/index.html#example-replica-exchange-wang-landau) for 
//! parallel Replica exchange Wang Landau

#![deny(missing_docs, warnings)]
#![allow(clippy::missing_safety_doc)]
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

pub mod metropolis;

pub use metropolis::*;

pub use wang_landau::*;
pub use entropic_sampling::*;

#[cfg(feature="replica_exchange")]
pub mod rewl;
#[cfg(feature="replica_exchange")]
pub use rewl::*;

#[cfg(feature="replica_exchange")]
pub mod rees;
#[cfg(feature="replica_exchange")]
pub use rees::*;


pub use histogram::*;
pub use heatmap::*;
pub use glue::*;
pub use traits::*;
#[cfg(feature="bootstrap")]
pub use bootstrap::*;

pub mod examples;