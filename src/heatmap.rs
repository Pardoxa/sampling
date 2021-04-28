//! # Generate heatmaps. Create Gnuplot scripts to plot said heatmaps
//! Note, you can use `HeatmapUsize` and `HeatmapF64` for **all** types, for which you have
//! implemented a histogram
mod heatmap;
mod helper;
mod gnuplot;
mod heatmap_float;
mod heatmap_u_mean;
mod heatmap_f_mean;

pub use heatmap::*;
pub use helper::*;
pub use gnuplot::*;
pub use heatmap_float::*;
pub use heatmap_u_mean::*;
pub use heatmap_f_mean::*;