use std::{borrow::Borrow, convert::From};
use crate::heatmap::{gnuplot_write_helper_plot, gnuplot_write_output};
use num_traits::AsPrimitive;
use crate::*;
use average::{MeanWithError, Estimate, WeightedMean};

pub struct HeatmapUsizeMean<HistX, HistY>
{
    pub(crate) heatmap: HeatmapUsize<HistX, HistY>,
    mean_with_errors: Vec<MeanWithError>
}

impl<HistX, HistY> HeatmapUsizeMean<HistX, HistY>
{
    pub fn heatmap(&self) -> &HeatmapU<HistX, HistY>
    {
        &self.heatmap
    }
}

impl<HistX, HistY> From<HeatmapUsize<HistX, HistY>> for HeatmapUsizeMean<HistX, HistY>
where HistX: Histogram,
    HistY: Histogram
{
    fn from(heatmap: HeatmapUsize<HistX, HistY>) -> Self {
        let x_bins = heatmap.hist_width.bin_count();
        let mean_with_errors = (0..x_bins)
            .map(|_| MeanWithError::new())
            .collect();

        Self{
            heatmap,
            mean_with_errors
        }
    }
}

impl<HistX, HistY> HeatmapUsizeMean<HistX, HistY>
where HistX: Histogram,
    HistY: Histogram,
{
    pub fn new(hist_x: HistX, hist_y: HistY) -> Self
    {
        let heatmap = HeatmapUsize::new(hist_x, hist_y);
        heatmap.into()
    }

    pub fn count_inside_heatmap<X, Y, A, B>(&mut self, x_val: A, y_val: B) -> Result<(usize, usize), HeatmapError>
    where HistX: HistogramVal<X>,
        HistY: HistogramVal<Y>,
        A: Borrow<X>,
        B: Borrow<Y>,
        Y: AsPrimitive<f64>
    {
        let x = x_val.borrow();
        let y = y_val.borrow();
        
        let res = self.heatmap.count(x, y);

        if let Ok((x, _)) = res {
            let y_f64 = y.as_();
            if y_f64.is_finite(){
                self.mean_with_errors[x].add(y_f64);
            }
        }
        res
    }

    pub fn count<X, Y, A, B>(&mut self, x_val: A, y_val: B) -> Result<(usize, usize), HeatmapError>
    where HistX: HistogramVal<X>,
        HistY: HistogramVal<Y>,
        A: Borrow<X>,
        B: Borrow<Y>,
        Y: AsPrimitive<f64>
    {
        let x = x_val.borrow();
        let y = y_val.borrow();

        let res = self.count_inside_heatmap(x, y);
        match res
        {
            Ok(_) => {},
            Err(_) => {
                let y_f64 = y.as_();
                if y_f64.is_finite() {
                    match self.heatmap.hist_width
                        .get_bin_index(x)
                    {
                        Ok(x_bin) => {
                            self.mean_with_errors[x_bin].add(y_f64);
                        },
                        _ => {}
                    }
                        
                }
            }
        }
        res
    }

    pub fn mean_vec(&self) -> &[MeanWithError]
    {
        &self.mean_with_errors
    }

    pub fn mean_iter<'a>(&'a self) -> impl Iterator<Item=f64> + 'a
    {
        self.mean_with_errors
            .iter()
            .map(
                |v|
                {
                    if v.is_empty(){
                        f64::NAN
                    } else 
                    {
                        v.mean()
                    }
                }
            )
    }

    pub fn mean(&self) -> Vec<f64>
    {
        let mut mean = Vec::with_capacity(self.mean_with_errors.len());

        mean.extend(self.mean_iter());
        mean
    }

    pub fn into_heatmap_normalized_columns(self) -> HeatmapF64Mean<HistX, HistY>
    {
        let heatmap = self.heatmap.into_heatmap_normalized_columns();
        let mut  mean = Vec::with_capacity(self.mean_with_errors.len());
        
        mean.extend(
            self.mean_with_errors.into_iter()
            .map(
                |v|
                {
                    if v.is_empty(){
                        WeightedMean::new()
                    }else {
                        let mut m = WeightedMean::new();
                        m.add(v.mean(), v.len() as f64);
                        m
                    }
                }
            )   
        );

        HeatmapF64Mean{
            heatmap,
            mean
        }
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
    /// I recommend that you take a look at [GnuplotSettings](crate::heatmap::GnuplotSettings)
    /// * `point_color`: the mean (in y-direction) will be plotted as points in the heatmap.
    /// Here you can choose the point color
    pub fn gnuplot<W, S, GS>(
        &self,
        mut writer: W,
        gnuplot_output_name: S,
        settings: GS,
        point_color: ColorRGB
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
        writeln!(writer, "$mean_data u 1:2:(1) pointtype 7 lc '0x000000' pointsize .57 notitle,\\")?;
        write!(writer, "$mean_data u 1:2:(1) pt 7 lc \"")?;
        point_color.write_hex(&mut writer)?;
        writeln!(writer, "\" ps 0.5 notitle")?;

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

        let mut heatmap_mean = HeatmapUsizeMean::new(hist_x, hist_y);
    
        for x in 0..=10 {
            for y in 0..=10{
                heatmap_mean.count_inside_heatmap(x, y).unwrap();
            }
        }

        for i in heatmap_mean.mean_iter() {
            assert_eq!(i, 5.0);
        }
    }

}