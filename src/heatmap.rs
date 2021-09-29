//! # Generate heatmaps. Create Gnuplot scripts to plot said heatmaps
//! Note, you can use [`HeatmapUsize`](crate::heatmap::HeatmapUsize) 
//! and [`HeatmapF64`](crate::heatmap::HeatmapF64) for **all** types, for which you have
//! implemented a histogram
//! # Examples
//! Here a simple Heatmap is created. 
//! I am using random data for the heatmap, to show how it works.
//! 
//! **Note**: both the x and y axis will be going from 0 to 10,
//! even though the y axis should go from 10 to 20.
//! The reason is, that the Histograms, which the Heatmap is based upon
//! do not have to be numbers at all. Therefore the default axis 
//! are the bin indices.
//! ```
//! use sampling::{HistUsizeFast, HeatmapU, GnuplotSettings, GnuplotAxis, GnuplotPalette};
//! use rand_pcg::Pcg64;
//! use rand::{SeedableRng, distributions::{Uniform, Distribution}};
//! use std::{fs::File, io::BufWriter};
//! 
//! // random number generator for data creation
//! let mut rng = Pcg64::seed_from_u64(2972398345698734489);
//! let dist_x = Uniform::from(0..11); // creates random numbers between 0 and 10
//! let dist_y = Uniform::from(10..21); // creates random numbers between 10 and 20
//! 
//! // I am now creating two histograms, which will be 
//! // defining the x and y axis of the heatmap
//! let hist_x = HistUsizeFast::new_inclusive(0, 10)
//!     .unwrap();
//! let hist_y = HistUsizeFast::new_inclusive(10, 20)
//!     .unwrap();
//! // create the Heatmap
//! let mut heat = HeatmapU::new(hist_x, hist_y);
//! 
//! for _ in 0..10000 {
//!     let x = dist_x.sample(&mut rng);
//!     let y = dist_y.sample(&mut rng);
//!     // counting the values
//!     heat.count(x, y).unwrap();
//! }
//! 
//! // creating a file to store the gnuplot script in
//! let heatmap_file = File::create("HeatmapU01.gp")
//!     .expect("unable to create file");
//! let heatmap_writer = BufWriter::new(heatmap_file);
//! let mut settings = GnuplotSettings::default();
//! 
//! // creating the gnuplot script
//! heat.gnuplot(
//!     heatmap_writer, 
//!     "HeatmapU01",
//!     &settings
//! ).unwrap();
//! 
//! // for correct axis, you have to set them yourself,
//! // since the histograms do not even need to be numeric
//! let x_axis = GnuplotAxis::new(0.0, 10.0, 6);
//! let y_axis = GnuplotAxis::new(10.0, 20.0, 6);
//! settings.x_axis(x_axis)
//!     .y_axis(y_axis)
//! // you can also change the color space, if you like
//!     .palette(GnuplotPalette::PresetRGB);
//! 
//! // creating a file to store the gnuplot script in
//! let heatmap_file = File::create("HeatmapU02.gp")
//!     .expect("unable to create file");
//! let heatmap_writer = BufWriter::new(heatmap_file);
//!
//! heat.gnuplot(
//!     heatmap_writer, 
//!     "HeatmapU02",
//!     settings
//! ).unwrap();
//! ```
//! Now you can create the plots by calling
//! ```bash
//! gnuplot HeatmapU01.gp HeatmapU02.gp
//! ```
//! which will create `HeatmapU01.pdf` and  `HeatmapU02.pdf`
mod heatmap;
mod helper;
mod gnuplot;
mod heatmap_float;
#[cfg(feature="bootstrap")]
mod heatmap_u_mean;
#[cfg(feature="bootstrap")]
mod heatmap_f_mean;

pub use heatmap::*;
pub use helper::*;
pub use gnuplot::*;
pub use heatmap_float::*;
#[cfg(feature="bootstrap")]
pub use heatmap_u_mean::*;
#[cfg(feature="bootstrap")]
pub use heatmap_f_mean::*;