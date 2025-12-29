//! # Generate heatmaps. Create Gnuplot scripts to plot said heatmaps
//! Note, you can use [`HeatmapUsize`]
//! and [`HeatmapF64`] for **all** types, for which you have
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
//! use {
//!     sampling::{HistUsizeFast, HeatmapU, GnuplotSettings, GnuplotAxis, GnuplotPalette, GnuplotTerminal},
//!     rand_pcg::Pcg64,
//!     rand::{SeedableRng, distr::{Uniform, Distribution}},
//!     std::{fs::File, io::BufWriter}
//! };
//!
//! // random number generator for data creation
//! let mut rng = Pcg64::seed_from_u64(2972398345698734489);
//! let dist_x = Uniform::new(0, 11).unwrap(); // creates random numbers between 0 and 10
//! let dist_y = Uniform::new(10, 21).unwrap(); // creates random numbers between 10 and 20
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
//! settings.terminal(GnuplotTerminal::PDF("HeatmapU01".to_owned()));
//!
//! // creating the gnuplot script
//! heat.gnuplot(
//!     heatmap_writer,
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
//!     .palette(GnuplotPalette::PresetRGB)
//!     .terminal(GnuplotTerminal::PDF("HeatmapU02".to_owned()));
//!
//! // creating a file to store the gnuplot script in
//! let heatmap_file = File::create("HeatmapU02.gp")
//!     .expect("unable to create file");
//! let heatmap_writer = BufWriter::new(heatmap_file);
//!
//! heat.gnuplot(
//!     heatmap_writer,
//!     settings
//! ).unwrap();
//! ```
//! Now you can create the plots by calling
//! ```bash
//! gnuplot HeatmapU01.gp HeatmapU02.gp
//! ```
//! which will create `HeatmapU01.pdf` and  `HeatmapU02.pdf`
mod gnuplot;
#[allow(clippy::module_inception)]
mod heatmap;
mod heatmap_float;
mod helper;

mod heatmap_u_mean;

mod heatmap_f_mean;

pub use gnuplot::*;
pub use heatmap::*;
pub use heatmap_float::*;
pub use helper::*;

pub use heatmap_u_mean::*;

pub use heatmap_f_mean::*;
