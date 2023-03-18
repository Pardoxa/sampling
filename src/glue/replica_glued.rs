use {
    crate::{
        glue_helper::{
            log10_to_ln, 
            ln_to_log10, 
            subtract_max,
            GlueErrors,
            height_correction
        },
        histogram::*,
    },
    super::derivative::*,
    std::borrow::Borrow
};

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

/// # Which LogBase is being used/should be used?
/// 
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum LogBase{
    /// use base 10
    Base10,
    /// use base e
    BaseE
}

impl LogBase{
    fn norm_log_prob(&self, slice: &mut [f64]) -> f64
    {
        match self{
            Self::Base10 => norm_log10_prob(slice),
            Self::BaseE => norm_ln_prob(slice)
        }
    }
}

// TODO maybe rename struct?
/// # Result of the gluing
#[derive(Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Glued<Hist>
{
    /// Histogram that encapsulates that is 
    pub(crate) encapsulating_histogram: Hist,
    pub(crate) glued: Vec<f64>,
    pub(crate) aligned: Vec<Vec<f64>>,
    pub(crate) base: LogBase,
    pub(crate) alignment: Vec<usize>
}

impl<Hist> Glued<Hist>
{
    /// Create a new `ReplicaGlued<Hist>` instance without checking anything
    pub fn new_unchecked(
        encapsulating_histogram: Hist, 
        glued: Vec<f64>, 
        aligned: Vec<Vec<f64>>, 
        base: LogBase, 
        alignment: Vec<usize>
    ) -> Self
    {
        Self{
            alignment,
            aligned,
            base,
            glued,
            encapsulating_histogram
        }
    }

    /// # Returns Slice which represents the glued logarithmic probability density
    /// The base of the logarithm can be found via [`self.base()`](`Self::base`)
    pub fn glued(&self) -> &[f64]
    {
        &self.glued
    }

    /// # Get alignment slice
    /// * Mostly used for internal things
    pub fn aligned(&self) -> &[Vec<f64>]
    {
        &self.aligned
    }

    /// # Returns encapsulating Histogram
    pub fn encapsulating_hist(&self) -> &Hist
    {
        &self.encapsulating_histogram
    }

    /// Returns the current base of the contained logarithms
    pub fn base(&self) -> LogBase
    {
        self.base
    }

    /// Change from Base 10 to Base E or the other way round
    pub fn switch_base(&mut self)
    {
        match self.base
        {
            LogBase::Base10 => {
                log10_to_ln(&mut self.glued);
                self.aligned
                    .iter_mut()
                    .for_each(|interval| log10_to_ln( interval));
                self.base = LogBase::BaseE;
            },
            LogBase::BaseE => {
                ln_to_log10(&mut self.glued);
                self.aligned
                    .iter_mut()
                    .for_each(|interval| ln_to_log10( interval));
                self.base = LogBase::Base10;
            }
        }
    }

}

impl<T> Glued<HistogramFast<T>>
where T: HasUnsignedVersion + num_traits::PrimInt + std::fmt::Display,
    T::Unsigned: num_traits::Bounded + HasUnsignedVersion<LeBytes=T::LeBytes> 
    + num_traits::WrappingAdd + num_traits::ToPrimitive + std::ops::Sub<Output=T::Unsigned>
{
    /// # Write the ReplicaGlued in a human readable format
    /// * You probably want to use this ;)
    pub fn write<W: std::io::Write>(&self, mut writer: W) -> std::io::Result<()>
    {
        writeln!(writer, "#bin log_merged log_interval0 …")?;
        writeln!(writer, "#log: {:?}", self.base)?;

        let mut alinment_helper = self.alignment_helper();

        for (&log_prob, bin) in self.glued
            .iter()
            .zip(self.encapsulating_histogram.bin_iter())
        {
            write!(writer, "{} {:e}", bin, log_prob)?;
            for (i, counter) in alinment_helper.iter_mut().enumerate()
            {
                if *counter < 0 {
                    write!(writer, " NaN")?
                } else {
                    let val = self.aligned[i].get(*counter as usize);
                    match val {
                        Some(&v) => write!(writer, " {:e}", v)?,
                        None => write!(writer, " NaN")?
                    }
                }
                *counter += 1;
            }
            writeln!(writer)?;
        }
        Ok(())
    }
}
impl<T> Glued<HistogramFast<T>>
{
    fn alignment_helper(&self) -> Vec<isize>
    {
        let mut alinment_helper: Vec<_> = std::iter::once(0)
            .chain(
                self.alignment.iter()
                    .map(|&v| -(v as isize))
            ).collect();

        let mut sum = 0;
        alinment_helper.iter_mut()
            .for_each(
                |val| 
                {
                    let old = sum;
                    sum += *val;
                    *val += old;
                }
            );
        alinment_helper
    }

    /// # Write the normalized probability density function
    /// The function will be normalized by using the binsize 
    /// you specify (uniform binsize is assumed).
    /// 
    /// Then it will write the merged log probability as well as the intervals 
    /// into the writer you specified
    pub fn write_rescaled<W: std::io::Write>(
        &self,
        mut writer: W,
        bin_size: f64,
        starting_point: f64
    ) -> std::io::Result<()>
    {
        writeln!(writer, "#bin log_merged log_interval0 …")?;
        writeln!(writer, "#log: {:?}", self.base)?;

        let bin_size_recip = bin_size.recip();

        let rescale = match self.base {
            LogBase::BaseE => bin_size_recip.ln(),
            LogBase::Base10 => bin_size_recip.log10(),
        };

        let mut alinment_helper = self.alignment_helper();

        for (index, log_prob) in self.glued
            .iter()
            .map(|s| *s + rescale)
            .enumerate()
        {
            let bin = starting_point + index as f64 * bin_size;
            write!(writer, "{} {:e}", bin, log_prob)?;
            for (i, counter) in alinment_helper.iter_mut().enumerate()
            {
                if *counter < 0 {
                    write!(writer, " NaN")?
                } else {
                    let val = self.aligned[i].get(*counter as usize);
                    match val {
                        Some(&v) => 
                        {
                            let rescaled = v + rescale;
                            write!(writer, " {:e}", rescaled)?
                        },
                        None => write!(writer, " NaN")?
                    }
                }
                *counter += 1;
            }
            writeln!(writer)?;
        }
        Ok(())
    }
} 

pub(crate) fn calc_merge_points(alignment: &[usize], derivatives: &[Vec<f64>]) -> Vec<usize>
{
    derivatives.iter()
        .zip(derivatives[1..].iter())
        .zip(alignment.iter())
        .map(
            |((left, right), &align)|
            {
                (align..)
                    .zip(
                        left[align..].iter()
                        .zip(right.iter())
                    )
                    .map(
                        |(index, (&left, &right))|
                        {
                            (index, (left - right).abs())
                        }
                    ).fold( (usize::MAX, f64::INFINITY),
                        |a, b|
                        if a.1 < b.1 {
                            a
                        } else {
                            b
                        }
                    ).0
            }
        ).collect()
}

/// # Calculate the probability density function from overlapping intervals
/// `log_prob` is a vector of the logarithmic non-normalized probability densities
/// 
/// `hists` is a vector of the corresponding histograms
/// 
/// `LogBase`: Which base do the logarithmic probabilities have?
/// 
/// This uses a derviative merge, that works similar to: [derivative_merged_log_prob_and_aligned](crate::rees::ReplicaExchangeEntropicSampling::derivative_merged_log_prob_and_aligned)
/// 
/// The [ReplicaGlued] allows you to easily write the probability density function to a file
pub fn derivative_merged_and_aligned<H, Hist>(
    mut log_prob: Vec<Vec<f64>>,
    hists: Vec<H>,
    log_base: LogBase
) -> Result<Glued<Hist>, HistErrors>
where Hist: HistogramCombine + Histogram,
    H: Borrow<Hist>
{

    // get the log_probabilities - the walkers over the same intervals are merged
    log_prob
        .iter_mut()
        .for_each(
            |v| 
            {
                subtract_max(v);
            }
        );
    // get the derivative, for merging later
    let derivatives: Vec<_> = log_prob.iter()
        .map(|v| derivative_merged(v))
        .collect();
    let e_hist = Hist::encapsulating_hist(&hists)?;
    let alignment  = hists.iter()
        .zip(hists.iter().skip(1))
        .map(|(left, right)| left.borrow().align(right.borrow()))
        .collect::<Result<Vec<_>, _>>()?;
    
    
    let merge_points = calc_merge_points(&alignment, &derivatives);

    // Not even one Interval - this has to be an error
    if log_prob.is_empty() {
        return Err(HistErrors::EmptySlice)
    }

    let mut merged_log_prob = vec![f64::NAN; e_hist.bin_count()];

    
    // Nothing to align, only one interval here
    if alignment.is_empty() {
        log_base.norm_log_prob(&mut log_prob[0]);
        
        let merged_prob = log_prob[0].clone();
        let r = 
            Glued::new_unchecked(
                e_hist,
                merged_prob,
                log_prob,
                log_base,
                alignment
            );
        return Ok(r);
    }

    merged_log_prob[..=merge_points[0]]
        .copy_from_slice(&log_prob[0][..=merge_points[0]]);


    // https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=2dcb7b7a3be78397d34657ece42aa851
    let mut align_sum = 0;

    for ((&a, &mp), log_prob_p1) in alignment.iter()
        .zip(merge_points.iter())
        .zip(log_prob.iter_mut().skip(1))
    {
        let position_l = mp + align_sum;
        align_sum += a;
        let left = mp - a;

        let shift = merged_log_prob[position_l] - log_prob_p1[left];

        // shift
        log_prob_p1
            .iter_mut()
            .for_each(|v| *v += shift);

        merged_log_prob[position_l..]
            .iter_mut()
            .zip(log_prob_p1[left..].iter())
            .for_each(
                |(merge, &val)|
                {
                    *merge = val;
                }
            );


    }

    let shift = log_base.norm_log_prob(&mut merged_log_prob);

    log_prob.iter_mut()
        .for_each(
            |interval|
            interval.iter_mut()
                .for_each(|val| *val -= shift)
        );

    let glued = 
        Glued::new_unchecked(
            e_hist, 
            merged_log_prob, 
            log_prob, 
            LogBase::BaseE, 
            alignment
        );
        
    Ok(
        glued
    )
}

pub(crate) fn norm_ln_prob(ln_prob: &mut[f64]) -> f64
{
    let max = subtract_max(ln_prob);
    // calculate actual sum in non log space
    let sum = ln_prob.iter()
        .fold(0.0, |acc, &val| {
            if val.is_finite(){
               acc +  val.exp()
            } else {
                acc
            }
        }
    );

    let shift = sum.ln();

    ln_prob.iter_mut()
        .for_each(|val| *val -= shift);

    shift - max

}

pub(crate) fn norm_log10_prob(log10_prob: &mut[f64]) -> f64
{
    let max = subtract_max(log10_prob);
    // calculate actual sum in non log space
    let sum = log10_prob.iter()
        .fold(0.0, |acc, &val| {
            if val.is_finite(){
               acc +  10.0_f64.powf(val)
            } else {
                acc
            }
        }
    );

    let shift = sum.log10();

    log10_prob.iter_mut()
        .for_each(|val| *val -= shift);

    shift - max

}

/// # Calculate the probability density function from overlapping intervals
/// `log_prob` is a vector of the logarithmic non-normalized probability densities
/// 
/// `hists` is a vector of the corresponding histograms
/// 
/// `LogBase`: Which base do the logarithmic probabilities have?
/// 
/// This uses a average merge, which first align all intervals and then merges 
/// the probability densities by averaging in the logarithmic space
/// 
/// The [ReplicaGlued] allows you to easily write the probability density function to a file
pub fn average_merged_and_aligned<Hist, H>(
    mut log_prob: Vec<Vec<f64>>,
    hists: Vec<H>,
    log_base: LogBase
) -> Result<Glued<Hist>, HistErrors>
where Hist: HistogramCombine + Histogram,
    H: Borrow<Hist>
{
    // get the log_probabilities - the walkers over the same intervals are merged
    log_prob
        .iter_mut()
        .for_each(
            |v| 
            {
                subtract_max(v);
            }
        );
    let e_hist = Hist::encapsulating_hist(&hists)?;
    let alignment  = hists.iter()
        .zip(hists.iter().skip(1))
        .map(|(left, right)| left.borrow().align(right.borrow()))
        .collect::<Result<Vec<_>, _>>()?;

    if alignment.is_empty(){
        // entering this means we only have 1 interval!
        assert_eq!(log_prob.len(), 1);
        log_base.norm_log_prob(&mut log_prob[0]);
        
        let glued = log_prob[0].clone();
        return Ok(
                Glued{
                base: LogBase::Base10,
                encapsulating_histogram: e_hist,
                aligned: log_prob,
                glued,
                alignment
            }
        );
    }

    // calc z
    let z_vec = calc_z(&log_prob, &alignment).expect("Unable to calculate Z in glueing");

    // correct height
    height_correction(&mut log_prob, &z_vec);
    // renaming
    let mut aligned_intervals = log_prob;

    // glueing together
    let mut glued_log_density = glue_no_derive(e_hist.bin_count(), &aligned_intervals, &alignment)
        .expect("Glue error!");

    // now norm the result
    let shift = log_base.norm_log_prob(&mut glued_log_density);

    aligned_intervals
        .iter_mut()
        .flat_map(|vec| vec.iter_mut())
        .for_each(|v| *v -= shift);

    Ok(
        Glued{
            base: log_base,
            encapsulating_histogram: e_hist,
            aligned: aligned_intervals,
            glued: glued_log_density,
            alignment
        }
    )
}

// TODO DOcument function, maybe rename function?
fn glue_no_derive(size: usize, log10_vec: &[Vec<f64>], alignment: &[usize]) -> Result<Vec<f64>, GlueErrors>
{
    let mut glue_log_density = vec![f64::NAN; size];


    // init - first interval can be copied for better performance
    let first_log = match log10_vec.first(){
        Some(interval) => interval.as_slice(),
        None => return Err(GlueErrors::EmptyList)
    };
   
    glue_log_density[0..first_log.len()].copy_from_slice(first_log);
    let mut glue_count = vec![0_usize; glue_log_density.len()];
    
    #[allow(clippy::needless_range_loop)]
    for (index, val) in first_log.iter().enumerate() {
        if val.is_finite(){
            glue_count[index] = 1;
        }
    }

    let mut offset = 0;
    for (i, log_vec) in log10_vec.iter().enumerate().skip(1)
    {
        offset += alignment[i-1];

        glue_log_density.iter_mut()
            .zip(glue_count.iter_mut())
            .skip(offset)
            .zip(log_vec.iter())
            .for_each(
                |((glued, count), &prob)|
                {
                    if prob.is_finite(){
                        *count += 1;
                        *glued = if glued.is_finite(){
                            *glued + prob
                        } else {
                            prob
                        };
                    }
                    
                }
            );
    }

    glue_log_density.iter_mut()
        .zip(glue_count.iter())
        .for_each(|(log, &count)| {
            if count > 0 {
                *log /= count as f64;
            } else {
                *log = f64::NAN;
            }
        });
    
    Ok(glue_log_density)
}

// TODO maybe rename function?
fn calc_z(log10_vec: &[Vec<f64>], alignment: &[usize]) -> Result<Vec<f64>, GlueErrors>
{
    let mut z_vec = Vec::with_capacity(alignment.len());
    for (i, &align) in alignment.iter().enumerate()
    {
        let prob_right = &log10_vec[i+1];
        let prob_left = &log10_vec[i][align..];
        
        let mut counter: usize = 0;
        let mut sum = 0.0;
        for (p, c) in prob_left.iter().zip(prob_right.iter())
        {
            if p.is_finite() && c.is_finite(){
                counter += 1;
                sum += p - c;
            }
        }
        
        let mut z = sum / counter as f64;
        // also correct for adjustment of prev
        if let Some(val) = z_vec.last() {
            z += val;
        }
        z_vec.push(z);
    }
    Ok(z_vec)
}