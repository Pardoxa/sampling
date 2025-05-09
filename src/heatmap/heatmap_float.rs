use{
    crate::*,
    std::{
        io::Write,
        borrow::*,
        convert::*
    },
    transpose::*
};

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

/// # Heatmap
/// * stores heatmap in row-major order: the rows of the heatmap are contiguous,
///   and the columns are strided
/// * enables you to quickly create a heatmap
/// * you can create gnuplot scripts to plot the heatmap
/// * you can transpose the heatmap
/// * …
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct HeatmapF64<HistWidth, HistHeight>{
    pub(crate) hist_width: HistWidth,
    pub(crate) hist_height: HistHeight,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) heatmap: Vec<f64>, // stored width, height
    pub(crate) error_count: usize
}

impl<HistWidth, HistHeight> From<HeatmapU<HistWidth, HistHeight>> for HeatmapF64<HistWidth, HistHeight>
where 
    HistWidth: Histogram,
    HistHeight: Histogram,
{
    fn from(other: HeatmapU<HistWidth, HistHeight>) -> Self {
        let mut heatmap = Vec::with_capacity(other.heatmap().len());
        heatmap.extend(
            other.heatmap()
                .iter()
                .map(|&val| val as f64)
        );
        Self{
            heatmap,
            width: other.width,
            height: other.height,
            hist_width: other.hist_width,
            hist_height: other.hist_height,
            error_count: other.error_count,
        }
    }
}

impl <HistWidth, HistHeight> HeatmapF64<HistWidth, HistHeight>
where 
    HistWidth: Clone,
    HistHeight: Clone,
{
    /// # Use this to get a "flipped" heatmap
    /// * creates a transposed heatmap
    /// * also look at [`self.transpose_inplace`](#method.transpose_inplace)
    pub fn transpose(&self) -> HeatmapF64<HistHeight, HistWidth>
    {
        let mut transposed = vec![0.0; self.heatmap.len()];
        transpose(
            &self.heatmap,
            &mut transposed,
            self.width,
            self.height
        );
        HeatmapF64{
            hist_width: self.hist_height.clone(),
            hist_height: self.hist_width.clone(),
            width: self.height,
            height: self.width,
            error_count: self.error_count,
            heatmap: transposed,
        }
    }
}

impl <HistWidth, HistHeight> HeatmapF64<HistWidth, HistHeight>
{

    /// # Use this to get a "flipped" heatmap
    /// * transposes the heatmap inplace
    pub fn transpose_inplace(mut self) -> HeatmapF64<HistHeight, HistWidth>
    {
        let mut scratch = vec![0.0; self.width.max(self.height)];
        transpose_inplace(&mut self.heatmap, &mut scratch, self.width, self.height);
        HeatmapF64{
            hist_width: self.hist_height,
            hist_height: self.hist_width,
            width: self.height,
            height: self.width,
            error_count: self.error_count,
            heatmap: self.heatmap
        }
    }

    /// x = j
    /// y = i
    #[inline(always)]
    fn index(&self, x: usize, y: usize) -> usize
    {
        heatmap_index(self.width, x, y)
    }

    /// Returns value stored in the heatmap at specified 
    /// coordinates, or `None`, if out of Bounds
    pub fn get(&self, x: usize, y: usize) -> Option<f64>
    {
        self.heatmap.get(self.index(x, y)).copied()
    }

    /// # row of the heatmap
    /// * `None` if out of bounds
    /// * otherwise it is a slice of the row at height `y`
    /// # Note
    /// *  there is no `get_column` method, because, due to implementation details,
    ///    it is way less efficient, and could not be returned as slice
    pub fn get_row(&self, y: usize) -> Option<&[f64]>
    {
        let fin = self.index(self.width, y);
        if fin > self.heatmap.len(){
            None
        } else {
            let start = self.index(0, y);
            Some(
                &self.heatmap[start..fin]
            )
        }
    }

    /// # row of the heatmap
    /// * returns reference of Slice of the specified row of the heatmap without checking for bounds 
    /// * Generally not recommended, use with caution! 
    /// ## Safety 
    /// Calling this with out-of-bounds index will result in undefined behavior!
    pub unsafe fn get_row_unchecked(&self, y: usize) -> &[f64]
    {
        let fin = self.index(self.width, y);
        let start = self.index(0, y);
        self.heatmap.get_unchecked(start..fin)
    }

    /// Returns value stored in the heatmap at specified 
    /// coordinates without performing bound checks.
    /// ## Safety
    /// **undefined behavior** if coordinates are out of bounds
    pub unsafe fn get_unchecked(&self, x: usize, y: usize) -> f64
    {
        *self.heatmap.get_unchecked(self.index(x, y))
    }

    /// # returns width of the heatmap
    /// * the width is the same size, as the `self.width_projection().bin_count()` 
    pub fn width(&self) -> usize
    {
        self.width
    }

    /// # returns height of the heatmap
    /// * the height is the same size, as the `self.height_projection().bin_count()` 
    pub fn height(&self) -> usize
    {
        self.height
    } 


    /// # Returns reference to current width Histogram
    /// * statistical information of how often a count hit a specific width
    pub fn width_count_hist(&self) -> &HistWidth{
        &self.hist_width
    }

    /// # Returns reference to current height Histogram
     /// * statistical information of how often a count hit a specific height
    pub fn height_count_hist(&self) -> &HistHeight{
        &self.hist_height
    }

    /// # Returns reference to current width Histogram
    /// * histogram used to bin in the "width" direction
    /// * all `counts` are counted here -> this is a projection of the heatmap
    pub fn width_hist(&self) -> &HistWidth{
        &self.hist_width
    }

    /// # Returns reference to current height Histogram
    /// * histogram used to bin in the "height" direction
    /// * all `counts` are counted here -> this is a projection of the heatmap
    pub fn height_hist(&self) -> &HistHeight{
        &self.hist_height
    }
}


impl<HistWidth, HistHeight> HeatmapF64<HistWidth, HistHeight>
where 
    HistWidth: Histogram,
    HistHeight: Histogram,
{

    /// # Create a new Heatmap
    /// * heatmap will have width `width_hist.bin_count()` 
    ///   and height `height_hist.bin_count()`
    /// * histograms will be reset (zeroed) here, so it does not matter, if they 
    ///   were used before and contain Data
    pub fn new(mut width_hist: HistWidth, mut height_hist: HistHeight) -> Self {
        let width = width_hist.bin_count();
        let height = height_hist.bin_count();
        width_hist.reset();
        height_hist.reset();
        let heatmap = vec![0.0; width * height];
        Self{
            width,
            height,
            heatmap,
            hist_width: width_hist,
            hist_height: height_hist,
            error_count: 0
        }
    }

    /// # Reset
    /// * resets histograms 
    /// * heatmap is reset to contain only 0's
    /// * miss_count is reset to 0
    pub fn reset(&mut self)
    {
        self.hist_width.reset();
        self.hist_height.reset();
        self.heatmap.iter_mut().for_each(|v| *v = 0.0);
        self.error_count = 0;
    }

    /// # "combine" heatmaps
    /// * heatmaps have to have the same dimensions
    /// * miss counts of other will be added to self
    /// * with and hight histogram counts will be added to self
    /// * `self.heatmap` will be modified at each index by 
    ///   `self.heatmap[i] = combine_fn(self.heatmap[i], other.heatmap[i])`
    /// # Usecase
    /// * e.g. if you want to add, subtract or multiply two heatmaps
    pub fn combine<OtherHW, OtherHH, F>
    (
        &mut self,
        other: &HeatmapF64<OtherHW, OtherHH>,
        combine_fn: F
    ) -> Result<(), HeatmapError>
    where OtherHW: Histogram,
        OtherHH: Histogram,
        F: Fn(f64, f64) -> f64
    {
        if self.width != other.width || self.height != other.height
        {
            return Err(HeatmapError::Dimension);
        }
        self.heatmap
            .iter_mut()
            .zip(
                other.heatmap.iter()
            ).for_each(
                |(this, &other)|
                {
                    *this = combine_fn(*this, other);
                }
            );
        
        for (i, &count) in other.hist_width.hist().iter().enumerate()
        {
            self.hist_width
                .increment_index_by(i, count)
                .unwrap()
        }

        for (i, &count) in other.hist_height.hist().iter().enumerate()
        {
            self.hist_height
                .increment_index_by(i, count)
                .unwrap()
        }
        self.error_count += other.error_count;
        
        Ok(())
    }

    /// # counts how often the heatmap was hit
    /// * should be equal to `self.heatmap.iter().sum::<usize>()` but more efficient
    /// * Note: it calculates this in O(min(self.width, self.height))
    pub fn total(&self) -> usize {
        if self.width <= self.height {
            self.hist_width.hist().iter().sum()
        } else {
            self.hist_height.hist().iter().sum()
        }
    }

    /// # Counts how often the Heatmap was missed, i.e., you tried to count a value (x,y), which was outside the Heatmap
    pub fn total_misses(&self) -> usize
    {
        self.error_count
    }


    /// # returns heatmap
    /// * each vector entry will contain the number of times, the corresponding bin was hit
    /// * an entry is 0 if it was never hit
    /// # Access indices; understanding how the data is mapped
    /// * A specific heatmap location `(x,y)`
    ///   corresponds to the index `y * self.width() + x`
    /// * you can use the `heatmap_index` function to calculate the index
    pub fn heatmap(&self) -> &Vec<f64>
    {
        &self.heatmap
    }

    /// # Normalizes self
    /// * Afterwards sum over all entries (within numerical precision) should be 1.0
    pub fn normalize_total(&mut self)
    {
        let sum = self.heatmap.iter().sum::<f64>();
        
        self.heatmap
            .iter_mut()
            .for_each(|val| *val /= sum);
    }

    /// # Normalizes self
    /// * Afterwards the sum of each column (fixed x) will be 1.0, if the sum of the row was not 0.0 before
    ///   If it did not, the column will only consist of 0.0
    pub fn normalize_columns(&mut self)
    {

        for x in 0..self.width {
            let denominator: f64 = (0..self.height)
                .map(|y| unsafe{self.get_unchecked(x, y)})
                .sum();

            if denominator != 0.0 {
                for y in 0..self.height {
                    let index = self.index(x, y);
                    unsafe {
                        *self.heatmap.get_unchecked_mut(index) /= denominator;
                    }
                }
            }
        }
    }

    /// # Normalizes self
    /// * Afterwards the sum of each row (fixed y) will be 1.0, if the sum of the row was not 0.0 before
    pub fn heatmap_normalize_rows(&mut self)
    {
        for y in 0..self.height {
            let row_sum = unsafe {self.get_row_unchecked(y).iter().sum::<f64>()};

            if row_sum != 0.0 {
                let index = self.index(0, y);
                for i in index..index + self.width {
                    unsafe {
                        *self.heatmap.get_unchecked_mut(i) /=  row_sum;
                    }
                }
            }
        }
    }
}

impl<HistWidth, HistHeight> HeatmapF64<HistWidth, HistHeight>
where 
    HistWidth: Histogram,
    HistHeight: Histogram,

{
    /// # update the heatmap
    /// * calculates the coordinate `(x, y)` of the bin corresponding
    ///   to the given value pair `(width_val, height_val)`
    /// * if coordinate is out of bounds, it counts a "miss" and returns the HeatmapError
    /// * otherwise it counts the "hit" (by adding `val` to the heatmap at the corresponding location)
    ///   and returns the coordinate `(x, y)` of the hit 
    pub fn count<A, B, X, Y>(&mut self, width_val: A, height_val: B, val: f64) -> Result<(usize, usize), HeatmapError>
    where 
        HistWidth: HistogramVal<X>,
        HistHeight: HistogramVal<Y>,
        A: Borrow<X>,
        B: Borrow<Y>
    {
        let x = self.hist_width
            .get_bin_index(width_val)
            .map_err(|e| {
                    self.error_count += 1;
                    HeatmapError::XError(e)
                }
            )?;
        let y = self.hist_height
            .count_val(height_val)
            .map_err(|e| {
                self.error_count += 1;
                HeatmapError::YError(e)
            }
        )?;
        
        let index = self.index(x, y);
        unsafe{
            *self.heatmap.get_unchecked_mut(index) += val;
        }

        self.hist_width
            .increment_index(x)
            .unwrap();

        Ok((x, y))
    }

    /// # Write heatmap to file
    /// * writes data of heatmap to file.
    /// # file
    /// * lets assume `self.width()`is 4 and `self.height()` is 3
    /// * the resulting file could look like
    /// ```txt
    /// 0.1 1.0 0.0 10.0
    /// 100.0 0.2 0.3 1.1
    /// 2.2 9.3 1.0 0.0
    /// ```
    pub fn write_to<W>(&self, mut data_file: W) -> std::io::Result<()>
    where W: Write
    {
        for y in 0..self.height {
            let row = unsafe{ self.get_row_unchecked(y) };

            if let Some((last, slice)) = row.split_last() {
                for val in slice {
                    write!(data_file, "{:e} ", val)?;
                }
                writeln!(data_file, "{:e}", last)?;
            }
        }
        Ok(())
    }

    /// # Create a gnuplot script to plot your heatmap
    /// * `writer`: The gnuplot script will be written to this
    /// # Note
    /// * This is the same as calling [`gnuplot`](Self::gnuplot) with default
    ///   `GnuplotSettings`
    /// * The default axis are the bin indices, which, e.g, means they always 
    ///   begin at 0. You have to set the axis via the [GnuplotSettings](crate::heatmap::GnuplotSettings)
    pub fn gnuplot_quick<W>(
        &self,
        writer: W
    ) -> std::io::Result<()>
    where 
        W: std::io::Write
    {
        let mut d = GnuplotSettings::default();
        let default = d
            .terminal(GnuplotTerminal::Empty);
        self.gnuplot(
            writer,
            default
        )
    }

    /// # Create a gnuplot script to plot your heatmap
    /// This function writes a file, that can be plotted via the terminal via [gnuplot](http://www.gnuplot.info/)
    /// ```bash
    /// gnuplot gnuplot_file
    /// ```
    /// ## Parameter:
    /// * `gnuplot_writer`: writer gnuplot script will be written to
    /// * `gnuplot_output_name`: how shall the file, created by executing gnuplot, be called? Ending of file will be set automatically
    /// ## Note
    /// * The default axis are the bin indices, which, e.g, means they always 
    ///   begin at 0. You have to set the axis via the [GnuplotSettings](crate::heatmap::GnuplotSettings)
    /// ## Example
    /// ```
    /// use rand_pcg::Pcg64;
    /// use rand::{SeedableRng, distr::*};
    /// use sampling::*;
    /// use std::fs::File;
    /// use std::io::BufWriter;
    /// 
    /// // first randomly create a heatmap
    /// let h_x = HistUsizeFast::new_inclusive(0, 10).unwrap();
    /// let h_y = HistU8Fast::new_inclusive(0, 10).unwrap();
    ///
    /// let mut heatmap = HeatmapU::new(h_x, h_y);
    /// heatmap.count(0, 0).unwrap();
    /// heatmap.count(10, 0).unwrap();
    ///
    /// let mut rng = Pcg64::seed_from_u64(27456487);
    /// let x_distr = Uniform::new_inclusive(0, 10_usize)
    ///     .unwrap();
    /// let y_distr = Uniform::new_inclusive(0, 10_u8)
    ///     .unwrap();
    ///
    /// for _ in 0..100000 {
    ///     let x = x_distr.sample(&mut rng);
    ///     let y = y_distr.sample(&mut rng);
    ///     heatmap.count(x, y).unwrap();
    /// }
    /// 
    /// // create File for gnuplot skript
    /// let file = File::create("heatmap_normalized.gp").unwrap();
    /// let buf = BufWriter::new(file);
    ///
    /// // Choose settings for gnuplot
    /// let mut settings = GnuplotSettings::new();
    /// settings.x_axis(GnuplotAxis::new(-5.0, 5.0, 6))
    ///     .y_axis(GnuplotAxis::from_slice(&["a", "b", "c", "d"]))
    ///     .y_label("letter")
    ///     .x_label("number")
    ///     .title("Example")
    ///     .terminal(GnuplotTerminal::PDF("heatmap_normalized".to_owned()));
    ///
    /// 
    /// // norm heatmap row wise - this converts HeatmapU to HeatmapfF64
    /// let heatmap = heatmap.into_heatmap_normalized_rows();
    ///
    /// // create script
    /// heatmap.gnuplot(
    ///     buf,
    ///     settings
    /// ).unwrap();
    /// ```
    /// Script can now be plotted with
    /// ```bash
    /// gnuplot heatmap_normalized.gp
    /// ```
    pub fn gnuplot<W, GS>(
        &self,
        mut gnuplot_writer: W,
        settings: GS
    ) -> std::io::Result<()>
    where 
        W: Write,
        GS: Borrow<GnuplotSettings>
    {
        let settings: &GnuplotSettings = settings.borrow();
        let x_len = self.width;
        let y_len = self.height;
        settings.write_heatmap(
            &mut gnuplot_writer,
            |w| self.write_to(w),
            x_len,
            y_len
        )
    }

}

