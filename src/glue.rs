//! # Glue together overlapping intervals of either entropic sampling or wang landau
//! Use this, if it makes sense to split the range, you are interested in, into multiple intervals

pub(crate) mod glue_helper;
mod glue_writer;
mod glue_job;

/// # Module for numeric derivatives
/// * Mostly intended for internal use, but you may use the functions as well if they help you
pub mod derivative;

pub use glue_helper::{GlueErrors, norm_log10_sum_to_1};
pub use glue_writer::*;
pub use glue_job::*;