//! # For sampling ensembles
//! * contains Simple sampling, WangLandau, entropic sampling, Metropolis, Histograms
//!
//! * [Detailed example](examples/coin_flips/index.html) for Wang-landau with successive entropic sampling
//!   and comparison with analytical results
//! * [Detailed example](examples/coin_flips/index.html#example-replica-exchange-wang-landau) for
//!   parallel Replica exchange Wang Landau

// TODO Remove comment to force Documentation of everything as requirement for compiling
#![deny(missing_docs, warnings)]
/// Contains traits useful for sampling an ensemble
/// like MarkovChain or Metropolis etc.
pub mod traits;

pub mod entropic_sampling;
pub mod glue;
pub mod heatmap;
pub mod histogram;
pub mod wang_landau;

pub mod bootstrap;

pub mod metropolis;

pub use metropolis::*;

pub use entropic_sampling::*;
pub use wang_landau::*;

#[cfg(feature = "replica_exchange")]
pub mod rewl;
#[cfg(feature = "replica_exchange")]
pub use rewl::*;

#[cfg(feature = "replica_exchange")]
pub mod rees;
#[cfg(feature = "replica_exchange")]
pub use rees::*;

pub use glue::*;
pub use heatmap::*;
pub use histogram::*;
pub use traits::*;

pub use bootstrap::*;

pub mod examples;
