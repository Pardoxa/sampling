//! # Glue together overlapping intervals of either entropic sampling or wang landau
//! Use this, if it makes sense to split the range, you are interested in, into multiple intervals
mod glue_wl;
pub(crate) mod glue_helper;
mod glue_entropic;
mod replica_glued;

pub mod derivative;

pub use glue_wl::*;
pub use glue_entropic::*;
pub use glue_helper::{GlueErrors, norm_log10_sum_to_1};
pub use replica_glued::*;