use std::cmp::*;

#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

/// List of possible errors
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum WangLandauErrors {
    /// `trial_step_min <= trial_step_max` has to be true
    InvalidMinMaxTrialSteps,
    /// `log_f_threshold`can never be negative or zero!
    /// it also cannot be NaN or infinite!
    InvalidLogFThreshold,
    /// Still in the process of gathering statistics
    /// Not enough to make an estimate
    NotEnoughStatistics,
    /// Still Gathering Statistics, this is only an estimate!
    EstimatedStatistic(Vec<f64>),
    /// bestof has to be at least 1 and at most the number of distinct steps tried,
    /// i.e., max_step - min_step + 1
    InvalidBestof,

    /// check refine has to be at least 1
    CheckRefineEvery0,

    /// you have to call one of the
    NotInitialized,

    /// Step limit exceeded without finding valid starting point
    InitFailed,
}

/// Look at the paper
/// > R. E. Belardinelli and V. D. Pereyra,
/// > Fast algorithm to calculate density of states,”
/// > Phys.&nbsp;Rev.&nbsp;E&nbsp;**75**: 046701 (2007), DOI&nbsp;[10.1103/PhysRevE.75.046701](https://doi.org/10.1103/PhysRevE.75.046701)
/// * This enum is to see, which mode is currently used
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum WangLandauMode {
    /// * Using original wang landau, i.e.,
    ///   refine every time when every bin in the
    ///   histogram was hit
    /// * refine: `log_f *= 0.5;
    RefineOriginal,
    /// * Use 1/T approach
    /// * refine each step by: `log_f = bin_count as f64 / step_count as f64`
    Refine1T,
}

impl WangLandauMode {
    /// * true if self is `RefineOriginal` variant
    /// * false otherwise
    pub fn is_mode_original(&self) -> bool {
        matches!(self, WangLandauMode::RefineOriginal)
    }

    /// * true if self is `Refine1T` variant
    /// * false otherwise
    pub fn is_mode_1_t(&self) -> bool {
        matches!(self, WangLandauMode::Refine1T)
    }
}

pub(crate) struct ProbIndex {
    pub(crate) index: usize,
    diff: f64,
}

impl PartialEq for ProbIndex {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.diff == other.diff
    }
}

impl Eq for ProbIndex {}

impl PartialOrd for ProbIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ProbIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        self.diff.total_cmp(&other.diff)
    }
}

impl ProbIndex {
    pub(crate) fn new(prob: f64, index: usize) -> Self {
        debug_assert!(prob.is_finite());
        Self {
            index,
            diff: (0.5 - prob).copysign(-1.0),
        }
    }

    pub(crate) fn is_best_of(&self, threshold: f64) -> bool {
        self.diff >= -threshold
    }
}
