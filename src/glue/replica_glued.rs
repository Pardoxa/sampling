use crate::{log10_to_ln, ln_to_log10};
use crate::histogram::*;

// TODO Document enum
#[derive(Clone, Copy, Debug)]
pub enum LogBase{
    Base10,
    BaseE
}

// TODO maybe rename struct?
#[derive(Clone)]
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