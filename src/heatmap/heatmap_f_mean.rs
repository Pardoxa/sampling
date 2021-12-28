use{
    crate::*,
    average::WeightedMean,
    num_traits::AsPrimitive,
    std::borrow::Borrow
};

/// # Heatmap with mean of y-axis
/// * stores heatmap in row-major order: the rows of the heatmap are contiguous,
/// and the columns are strided
/// * enables you to quickly create a heatmap
/// * you can create gnuplot scripts to plot the heatmap
/// * for each x-axis bin, the y-axis mean is calculated
/// * …
/// 
/// # Difference to `HeatmapF64`
/// * [`HeatmapF64`](crate::heatmap::HeatmapF64) does not contain the averages for th y-axis,
/// but can be transposed and also used for Y-Histograms which take types which do not 
/// implement AsPrimitive<f64>
pub struct HeatmapF64Mean<HistX, HistY>
{
    pub(crate) heatmap: HeatmapF64<HistX, HistY>,
    pub(crate) mean: Vec<WeightedMean>
}

impl<HistX, HistY> HeatmapF64Mean<HistX, HistY>
{
    /// Internal [`HeatmapF64`](crate::heatmap::HeatmapF64)
    pub fn heatmap(&self) -> &HeatmapF64<HistX, HistY>
    {
        &self.heatmap
    }
}

impl<HistX, HistY> HeatmapF64Mean<HistX, HistY>
where HistX: Histogram,
    HistY: Histogram,
{
    /// # Create a heatmap
    /// * creates new instance
    /// * `hist_x` defines the bins along the x-axis
    /// * `hist_y` defines the bins along the y-axis
    pub fn new(hist_x: HistX, hist_y: HistY) -> Self
    {
        let heatmap = HeatmapF64::new(hist_x, hist_y);
        let x_bins = heatmap.hist_width.bin_count();
        let mean = (0..x_bins)
            .map(|_| WeightedMean::new())
            .collect();

        Self{
            heatmap,
            mean
        }
    }

    /// # Update Heatmap
    /// * similar to [`count` of `HeatmapF64`](crate::heatmap::HeatmapF64::count)
    /// 
    /// This time, however, any value that is out of bounds will be ignored for
    /// the calculation of the mean of the y-axis, meaning also values which correspond 
    /// to a valid x-bin will be ignored, if their y-value is not inside the Y Histogram.
    /// The mean respects the `weight`
    pub fn count_inside_heatmap<X, Y, A, B>(
        &mut self,
        x_val: A,
        y_val: B,
        weight: f64
    ) -> Result<(usize, usize), HeatmapError>
    where HistX: HistogramVal<X>,
        HistY: HistogramVal<Y>,
        A: Borrow<X>,
        B: Borrow<Y>,
        Y: AsPrimitive<f64>
    {
        let x = x_val.borrow();
        let y = y_val.borrow();
        
        let res = self.heatmap.count(x, y, weight);

        if let Ok((x, _)) = res {
            let y_f64 = y.as_();
            if y_f64.is_finite(){
                self.mean[x].add(y_f64, weight);
            }
        }
        res
    }

    /// # Update heatmap
    /// * Corresponds to [`count` of `HeatmapU`](crate::heatmap::HeatmapU::count)
    /// 
    /// The difference is, that the mean of the y-axis is updated as long as `y_val` is finite
    /// and `x_val` is in bounds (because the mean is calculated for each bin in the x direction
    /// separately). The calculated average respects the `weight`
    pub fn count<X, Y, A, B>(&mut self, x_val: A, y_val: B, weight: f64) -> Result<(usize, usize), HeatmapError>
    where HistX: HistogramVal<X>,
        HistY: HistogramVal<Y>,
        A: Borrow<X>,
        B: Borrow<Y>,
        Y: AsPrimitive<f64>
    {
        let x = x_val.borrow();
        let y = y_val.borrow();

        let res = self.count_inside_heatmap(x, y, weight);
        match res
        {
            Ok(_) => {},
            Err(_) => {
                let y_f64 = y.as_();
                if y_f64.is_finite() {
                    if let Ok(x_bin) = self
                        .heatmap
                        .hist_width
                        .get_bin_index(x)
                    {
                        self.mean[x_bin].add(y_f64, weight);
                    }   
                }
            }
        }
        res
    }

    /// # Internal slice for mean
    /// * The mean is calculated from this slice
    /// * The mean corresponds to the bins of the x-axis
    pub fn mean_slice(&self) -> &[WeightedMean]
    {
        &self.mean
    }

    /// # Iterate over the calculated mean
    /// * iterates over the means
    /// * The mean corresponds to the bins of the x-axis
    /// * if a bin on the x-axis has no entries, the corresponding
    /// mean will be `f64::NAN`
    pub fn mean_iter(&'_ self) -> impl Iterator<Item=f64> + '_
    {
        self.mean
            .iter()
            .map(
                |v|
                {
                    if v.is_empty(){
                        f64::NAN
                    } else {
                        v.mean()
                    }
                }
            )
    }

    /// # Get a mean vector
    /// * The entries are the means corresponds to the bins of the x-axis
    /// * if a bin on the x-axis has no entries, the corresponding
    /// mean will be `f64::NAN`
    /// 
    /// # Note
    /// * If you want to iterate over the mean values, use 
    /// [`mean_iter`](Self::mean_iter) instead
    pub fn mean(&self) -> Vec<f64>
    {
        let mut mean = Vec::with_capacity(self.mean.len());

        mean.extend(self.mean_iter());
        mean
    }

    /// # Create a gnuplot script to plot your heatmap
    /// * `writer`: The gnuplot script will be written to this
    /// * `gnuplot_output_name`: how shall the file, created by executing gnuplot, 
    /// be called? Ending of file will be set automatically
    /// # Note
    /// * This is the same as calling [`gnuplot`](Self::gnuplot) with default
    /// [`GnuplotSettings`](crate::heatmap::GnuplotSettings) and default 
    /// [`GnuplotPointSettings`](crate::heatmap::GnuplotPointSettings)
    /// * The default axis are the bin indices, which, e.g, means they always 
    /// begin at 0. You have to set the axis via the [GnuplotSettings](crate::heatmap::GnuplotSettings)
    pub fn gnuplot_quick<W, S>(
        &self,
        writer: W,
        gnuplot_output_name: S
    ) -> std::io::Result<()>
    where 
        W: std::io::Write,
        S: AsRef<str>
    {
        self.gnuplot(
            writer,
            gnuplot_output_name,
            GnuplotSettings::default(),
            GnuplotPointSettings::default()
        )
    }

    /// # Create a gnuplot script to plot your heatmap
    /// This function writes a file, that can be plotted in the terminal via [gnuplot](http://www.gnuplot.info/)
    /// ```bash
    /// gnuplot gnuplot_file
    /// ```
    /// ## Parameter:
    /// * `writer`: writer gnuplot script will be written to
    /// * `gnuplot_output_name`: how shall the file, created by executing gnuplot, be called? File suffix (ending) will be set automatically
    /// * `settings`: Here you can set the axis, choose between terminals and more. 
    /// I recommend that you take a look at [GnuplotSettings](crate::heatmap::GnuplotSettings)
    /// * `point_color`: the mean (in y-direction) will be plotted as points in the heatmap.
    /// Here you can choose the point color
    /// ## Notes
    /// The default axis are the bin indices, which, e.g, means they always 
    /// begin at 0. You have to set the axis via the [GnuplotSettings](crate::heatmap::GnuplotSettings)
    pub fn gnuplot<W, S, GS>(
        &self,
        mut writer: W,
        gnuplot_output_name: S,
        settings: GS,
        point: GnuplotPointSettings
    ) -> std::io::Result<()> 
    where 
        W: std::io::Write,
        S: AsRef<str>,
        GS: Borrow<GnuplotSettings>
    {
        let settings = settings.borrow();
        
        self.heatmap.gnuplot_write_helper_setup(
            &mut writer,
            gnuplot_output_name.as_ref(),
            settings
        )?;

        writeln!(writer, "$mean_data << EOD")?;
        for (index, value) in self.mean_iter().enumerate()
        {
            writeln!(writer, "{} {:e}", index, value)?;
        }
        writeln!(writer, "EOD")?;
        gnuplot_write_helper_plot(
            &mut writer,
            settings.get_title()
        )?;
        writeln!(writer, ",\\")?;
        if point.frame
        {
            write!(writer, "$mean_data u 1:2:(1) pointtype 7 lc \"")?;
            point.frame_color.write_hex(&mut writer)?;
            writeln!(writer, "\" pointsize {} notitle,\\", point.frame_size())?;
        }

        write!(writer, "$mean_data u 1:2:(1) pt 7 lc \"")?;
        point.color.write_hex(&mut writer)?;
        writeln!(writer, "\" ps {} t \"{}\"", point.get_size(), point.get_legend())?;
        

        gnuplot_write_output(
            writer,
            gnuplot_output_name.as_ref(),
            settings
        )

    } 

}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::HistUsizeFast;

    #[test]
    fn average_test()
    {
        let hist_x = HistUsizeFast::new_inclusive(0, 10)
            .unwrap();
        
        let hist_y = hist_x.clone();

        let mut heatmap_mean = HeatmapF64Mean::new(hist_x, hist_y);
    
        for x in 0..=10 {
            for y in 0..=10{
                heatmap_mean.count_inside_heatmap(x, y, 1.0).unwrap();
            }
        }

        for i in heatmap_mean.mean_iter() {
            assert_eq!(i, 5.0);
        }
    }

}