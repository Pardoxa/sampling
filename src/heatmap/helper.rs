use crate::*;

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
/// # Errors of Heatmap
pub enum HeatmapError{
    /// An Error while calculating the index of the x coordinate
    XError(HistErrors),
    /// An Error while calculating the index of the y coordinate
    YError(HistErrors),
    /// you tried to combine heatmaps of different Dimensions
    Dimension
}
