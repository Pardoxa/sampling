use {
    crate::*,
    average::{Estimate, MeanWithError, WeightedMean},
    num_traits::AsPrimitive,
    std::borrow::Borrow,
};

/// # Heatmap with mean of y-axis
/// * stores heatmap in row-major order: the rows of the heatmap are contiguous,
///   and the columns are strided
/// * enables you to quickly create a heatmap
/// * you can create gnuplot scripts to plot the heatmap
/// * for each x-axis bin, the y-axis mean is calculated
/// * …
///
/// # Difference to `HeatmapU`
/// * [`HeatmapU`](crate::heatmap::HeatmapU) does not contain the averages for th y-axis,
///   but can be transposed and also used for Y-Histograms which take types which do not
///   implement `AsPrimitive<f64>`
pub struct HeatmapUsizeMean<HistX, HistY> {
    pub(crate) heatmap: HeatmapUsize<HistX, HistY>,
    mean_with_errors: Vec<MeanWithError>,
}

impl<HistX, HistY> HeatmapUsizeMean<HistX, HistY> {
    /// Internal [`HeatmapU`](crate::heatmap::HeatmapU)
    pub fn heatmap(&self) -> &HeatmapU<HistX, HistY> {
        &self.heatmap
    }
}

impl<HistX, HistY> HeatmapUsizeMean<HistX, HistY>
where
    HistX: Histogram,
    HistY: Histogram,
{
    /// # Create a heatmap
    /// * creates new instance
    /// * `hist_x` defines the bins along the x-axis
    /// * `hist_y` defines the bins along the y-axis
    pub fn new(hist_x: HistX, hist_y: HistY) -> Self {
        let heatmap = HeatmapUsize::new(hist_x, hist_y);
        let x_bins = heatmap.hist_width.bin_count();
        let mean_with_errors = (0..x_bins).map(|_| MeanWithError::new()).collect();

        Self {
            heatmap,
            mean_with_errors,
        }
    }

    /// # Update Heatmap
    /// * similar to [`count` of `HeatmapU`](crate::heatmap::HeatmapUsizeMean::count)
    ///
    /// This time, however, any value that is out of bounds will be ignored for
    /// the calculation of the mean of the y-axis, meaning also values which correspond
    /// to a valid x-bin will be ignored, if their y-value is not inside the Y Histogram
    pub fn count_inside_heatmap<X, Y, A, B>(
        &mut self,
        x_val: A,
        y_val: B,
    ) -> Result<(usize, usize), HeatmapError>
    where
        HistX: HistogramVal<X>,
        HistY: HistogramVal<Y>,
        A: Borrow<X>,
        B: Borrow<Y>,
        Y: AsPrimitive<f64>,
    {
        let x = x_val.borrow();
        let y = y_val.borrow();

        let res = self.heatmap.count(x, y);

        if let Ok((x, _)) = res {
            let y_f64 = y.as_();
            if y_f64.is_finite() {
                self.mean_with_errors[x].add(y_f64);
            }
        }
        res
    }

    /// # Update heatmap
    /// * Corresponds to [`count` of `HeatmapU`](crate::heatmap::HeatmapU::count)
    ///
    /// The difference is, that the mean of the y-axis is updated as long as `y_val` is finite
    /// and `x_val` is in bounds (because the mean is calculated for each bin in the x direction
    /// separately)
    pub fn count<X, Y, A, B>(&mut self, x_val: A, y_val: B) -> Result<(usize, usize), HeatmapError>
    where
        HistX: HistogramVal<X>,
        HistY: HistogramVal<Y>,
        A: Borrow<X>,
        B: Borrow<Y>,
        Y: AsPrimitive<f64>,
    {
        let x = x_val.borrow();
        let y = y_val.borrow();

        let res = self.count_inside_heatmap(x, y);
        match res {
            Ok(_) => {}
            Err(_) => {
                let y_f64 = y.as_();
                if y_f64.is_finite() {
                    if let Ok(x_bin) = self.heatmap.hist_width.get_bin_index(x) {
                        self.mean_with_errors[x_bin].add(y_f64);
                    }
                }
            }
        }
        res
    }

    /// # Internal slice for mean
    /// * The mean is calculated from this slice
    /// * The mean corresponds to the bins of the x-axis
    /// * you can also access the estimated error of the mean here
    pub fn mean_slice(&self) -> &[MeanWithError] {
        &self.mean_with_errors
    }

    /// # Iterate over the calculated mean
    /// * iterates over the means
    /// * The mean corresponds to the bins of the x-axis
    /// * if a bin on the x-axis has no entries, the corresponding
    ///   mean will be `f64::NAN`
    pub fn mean_iter(&'_ self) -> impl Iterator<Item = f64> + '_ {
        self.mean_with_errors
            .iter()
            .map(|v| if v.is_empty() { f64::NAN } else { v.mean() })
    }

    /// # Get a mean vector
    /// * The entries are the means corresponds to the bins of the x-axis
    /// * if a bin on the x-axis has no entries, the corresponding
    ///   mean will be `f64::NAN`
    ///
    /// # Note
    /// * If you want to iterate over the mean values, use
    ///   [`mean_iter`](Self::mean_iter) instead
    /// * If you require error information, take a look at [`mean_slice`](Self::mean_slice)
    pub fn mean(&self) -> Vec<f64> {
        let mut mean = Vec::with_capacity(self.mean_with_errors.len());

        mean.extend(self.mean_iter());
        mean
    }

    /// # returns (column wise) normalized heatmap
    /// * returns normalized heatmap as [`HeatmapF64Mean`](crate::heatmap::HeatmapF64Mean)
    ///
    ///
    /// * Heatmap vector `self.heatmap_normalized().heatmap()` contains only 0.0, if nothing was in the heatmap
    /// * otherwise the sum of each column (fixed x) will be 1.0 (within numerical errors), if it contained at least one hit.
    ///   If it did not, the column will only consist of 0.0
    /// * otherwise the sum of this Vector is 1.0
    ///
    /// For the calculation of the mean, each `count` will have a weight of 1
    pub fn into_heatmap_normalized_columns(self) -> HeatmapF64Mean<HistX, HistY> {
        let heatmap = self.heatmap.into_heatmap_normalized_columns();
        let mut mean = Vec::with_capacity(self.mean_with_errors.len());

        mean.extend(self.mean_with_errors.into_iter().map(|v| {
            if v.is_empty() {
                WeightedMean::new()
            } else {
                let mut m = WeightedMean::new();
                m.add(v.mean(), v.len() as f64);
                m
            }
        }));

        HeatmapF64Mean { heatmap, mean }
    }

    /// # Create a gnuplot script to plot your heatmap
    /// * `writer`: The gnuplot script will be written to this
    /// * `gnuplot_output_name`: how shall the file, created by executing gnuplot,
    ///   be called? Ending of file will be set automatically
    /// # Note
    /// * This is the same as calling [`gnuplot`](Self::gnuplot) with default
    ///   [`GnuplotSettings`](crate::heatmap::GnuplotSettings)
    ///   and default [`GnuplotPointSettings`](crate::heatmap::GnuplotPointSettings)
    /// * The default axis are the bin indices, which, e.g, means they always
    ///   begin at 0. You have to set the axis via the [GnuplotSettings](crate::heatmap::GnuplotSettings)
    pub fn gnuplot_quick<W>(&self, writer: W) -> std::io::Result<()>
    where
        W: std::io::Write,
    {
        self.gnuplot(
            writer,
            GnuplotSettings::default(),
            GnuplotPointSettings::default(),
        )
    }

    /// # Create a gnuplot script to plot your heatmap
    /// This function writes a file, that can be plotted via the terminal via [gnuplot](http://www.gnuplot.info/)
    /// ```bash
    /// gnuplot gnuplot_file
    /// ```
    /// ## Parameter:
    /// * `writer`: writer gnuplot script will be written to
    /// * `gnuplot_output_name`: how shall the file, created by executing gnuplot, be called? Ending of file will be set automatically
    /// * `settings`: Here you can set the axis, choose between terminals and more.
    ///   I recommend that you take a look at [GnuplotSettings](crate::heatmap::GnuplotSettings)
    /// * `point_color`: the mean (in y-direction) will be plotted as points in the heatmap.
    ///   Here you can choose the point color
    /// ## Note
    /// * The default axis are the bin indices, which, e.g, means they always
    ///   begin at 0. You have to set the axis via the [GnuplotSettings](crate::heatmap::GnuplotSettings)
    pub fn gnuplot<W, P, GS>(&self, mut writer: W, settings: GS, points: P) -> std::io::Result<()>
    where
        W: std::io::Write,
        P: Borrow<GnuplotPointSettings>,
        GS: Borrow<GnuplotSettings>,
    {
        let settings: &GnuplotSettings = settings.borrow();
        let point: &GnuplotPointSettings = points.borrow();

        let x_len = self.heatmap.width;
        let y_len = self.heatmap.height;

        settings.write_heatmap_helper1(&mut writer, x_len, y_len)?;

        writeln!(writer, "$mean_data << EOD")?;
        for (index, value) in self.mean_iter().enumerate() {
            writeln!(writer, "{} {:e}", index, value)?;
        }
        writeln!(writer, "EOD")?;
        writeln!(writer, "$data << EOD")?;
        self.heatmap.write_to(&mut writer)?;
        writeln!(writer, "EOD")?;
        write!(
            writer,
            "splot $data matrix with image t \"{}\" ",
            settings.get_title()
        )?;
        writeln!(writer, ",\\")?;
        if point.frame {
            write!(writer, "$mean_data u 1:2:(1) pointtype 7 lc \"")?;
            point.frame_color.write_hex(&mut writer)?;
            writeln!(writer, "\" pointsize {} notitle,\\", point.frame_size())?;
        }

        write!(writer, "$mean_data u 1:2:(1) pt 7 lc \"")?;
        point.color.write_hex(&mut writer)?;
        writeln!(
            writer,
            "\" ps {} t \"{}\"",
            point.get_size(),
            point.get_legend()
        )?;

        settings.terminal.finish(writer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HistUsizeFast;

    #[test]
    fn average_test() {
        let hist_x = HistUsizeFast::new_inclusive(0, 10).unwrap();

        let hist_y = hist_x.clone();

        let mut heatmap_mean = HeatmapUsizeMean::new(hist_x, hist_y);

        for x in 0..=10 {
            for y in 0..=10 {
                heatmap_mean.count_inside_heatmap(x, y).unwrap();
            }
        }

        for i in heatmap_mean.mean_iter() {
            assert_eq!(i, 5.0);
        }
    }
}
