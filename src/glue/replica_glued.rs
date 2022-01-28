use {
    crate::{
        glue_helper::{
            log10_to_ln, 
            ln_to_log10, 
            subtract_max,
        },
        histogram::*,
    },
    super::derivative::*,
    std::borrow::Borrow
};

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

// TODO Document enum
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum LogBase{
    Base10,
    BaseE
}

// TODO maybe rename struct?
#[derive(Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct ReplicaGlued<Hist>
{
    pub(crate) encapsulating_histogram: Hist,
    pub(crate) glued: Vec<f64>,
    pub(crate) aligned: Vec<Vec<f64>>,
    pub(crate) base: LogBase,
    pub(crate) alignment: Vec<usize>
}

impl<Hist> ReplicaGlued<Hist>
{
    pub unsafe fn new_unchecked(
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

    pub fn aligned(&self) -> &[Vec<f64>]
    {
        &self.aligned
    }

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

impl<T> ReplicaGlued<HistogramFast<T>>
where T: HasUnsignedVersion + num_traits::PrimInt + std::fmt::Display,
    T::Unsigned: num_traits::Bounded + HasUnsignedVersion<LeBytes=T::LeBytes> 
    + num_traits::WrappingAdd + num_traits::ToPrimitive + std::ops::Sub<Output=T::Unsigned>
{
    pub fn write<W: std::io::Write>(&self, mut writer: W) -> std::io::Result<()>
    {
        writeln!(writer, "#bin log_merged log_interval0 …")?;
        writeln!(writer, "#log: {:?}", self.base)?;

        let mut alinment_helper = self.alinment_helper();

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
impl<T> ReplicaGlued<HistogramFast<T>>
{
    fn alinment_helper(&self) -> Vec<isize>
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

        let mut alinment_helper = self.alinment_helper();

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

// TODO maybe rename function
#[allow(clippy::type_complexity)]
pub(crate) fn average_merged_log_probability_helper2<Hist>(
    mut log_prob: Vec<Vec<f64>>,
    hists: Vec<&Hist>
) -> Result<(Vec<usize>, Vec<Vec<f64>>, Hist), HistErrors>
where Hist: HistogramCombine
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
        .map(|(&left, &right)| left.align(right))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(
        (
            alignment,
            log_prob,
            e_hist
        )
    )
}



pub fn derivative_merged_and_aligned<H, Hist>(
    mut log_prob: Vec<Vec<f64>>,
    hists: Vec<H>
) -> Result<ReplicaGlued<Hist>, HistErrors>
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
        norm_ln_prob(&mut log_prob[0]);
        let merged_prob = log_prob[0].clone();
        let r = unsafe{   
            ReplicaGlued::new_unchecked(
                e_hist,
                merged_prob,
                log_prob,
                LogBase::BaseE,
                alignment
            )
        };
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
    
    let shift = norm_ln_prob(&mut merged_log_prob);
    log_prob.iter_mut()
        .for_each(
            |interval|
            interval.iter_mut()
                .for_each(|val| *val -= shift)
        );

    let glued = unsafe{
        ReplicaGlued::new_unchecked(
            e_hist, 
            merged_log_prob, 
            log_prob, 
            LogBase::BaseE, 
            alignment
        )
    };
        
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