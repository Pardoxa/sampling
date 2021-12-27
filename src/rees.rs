//! # Entropic sampling using a replica exchange approach
//! 
//! The entropic sampling itself is identical to the one from 
//! [this module](`crate::entropic_sampling::EntropicSampling`)
//! with the difference being, that there will be replica exchanges between
//! neighboring intervals (and if you have multiple walker in an interval, they can exchange states as well)
//!
//! This is intended to improve the sampling of difficult "energy landscapes"
#[allow(clippy::module_inception)]
mod rees;
mod walker;
mod merge;

pub use rees::*;
pub use walker::*;
pub(crate) use merge::*;