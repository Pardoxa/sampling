use{
    crate::{*, glue_helper::*},
    rayon::prelude::*,
    std::f64::consts::LOG10_E
};

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

pub(crate) fn only_merged<Hist>(
    merge_points: Vec<usize>,
    alignment: Vec<usize>,
    log_prob: Vec<Vec<f64>>,
    e_hist: Hist
) -> (Vec<f64>, Hist)
where Hist: Histogram
{
    let mut merged_log_prob = vec![f64::NAN; e_hist.bin_count()];

    if merge_points.is_empty() {
        // Nothing to merge - only one interval present
        merged_log_prob = log_prob.into_iter().next().unwrap();
        norm_ln_prob(&mut merged_log_prob);
        return (merged_log_prob, e_hist);
    }
    
    merged_log_prob[..=merge_points[0]]
        .copy_from_slice(&log_prob[0][..=merge_points[0]]);


    // https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=2dcb7b7a3be78397d34657ece42aa851
    let mut align_sum = 0;
    for (index, (&a, &mp)) in alignment.iter().zip(merge_points.iter()).enumerate()
    {
        let position_l = mp + align_sum;
        align_sum += a;
        let left = mp - a;

        let shift = merged_log_prob[position_l] - log_prob[index + 1][left];

        merged_log_prob[position_l..]
            .iter_mut()
            .zip(log_prob[index + 1][left..].iter())
            .for_each(
                |(merge, val)|
                {
                    *merge = val + shift;
                }
            );


    }
    norm_ln_prob(&mut merged_log_prob);

    (merged_log_prob, e_hist)
}

pub(crate) fn merged_and_aligned<'a, Hist: 'a, I>(
    hists: I,
    merge_points: Vec<usize>,
    alignment: Vec<usize>,
    log_prob: Vec<Vec<f64>>,
    e_hist: Hist
) -> GluedResult<Hist>
where Hist: HistogramCombine + Histogram,
    I: Iterator<Item = &'a Hist>
{
    // Not even one Interval - this has to be an error
    if log_prob.is_empty() {
        return Err(HistErrors::EmptySlice);
    }
    let mut merged_log_prob = vec![f64::NAN; e_hist.bin_count()];

    let mut aligned_intervals = vec![merged_log_prob.clone(); alignment.len() + 1];

    aligned_intervals[0][..log_prob[0].len()]
        .copy_from_slice(&log_prob[0]);
    
    // Nothing to allign, only one interval here
    if merge_points.is_empty() {
        norm_ln_prob(&mut aligned_intervals[0]);
        merged_log_prob.copy_from_slice(&aligned_intervals[0]);
        return Ok(
            (e_hist, merged_log_prob, aligned_intervals)
        );
    }

    merged_log_prob[..=merge_points[0]]
        .copy_from_slice(&log_prob[0][..=merge_points[0]]);


    // https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=2dcb7b7a3be78397d34657ece42aa851
    let mut align_sum = 0;

    for ((index, (&a, &mp)), hist) in alignment.iter()
        .zip(merge_points.iter()).enumerate()
        .zip(
            hists.skip(1)
        )
    {
        let position_l = mp + align_sum;
        align_sum += a;
        let left = mp - a;

        let index_p1 = index + 1; 

        let shift = merged_log_prob[position_l] - log_prob[index_p1][left];

        let unmerged_align = e_hist.align(hist)?;

        aligned_intervals[index_p1][unmerged_align..]
            .iter_mut()
            .zip(log_prob[index_p1].iter())
            .for_each(|(v, &val)| *v = val + shift);

        merged_log_prob[position_l..]
            .iter_mut()
            .zip(log_prob[index_p1][left..].iter())
            .for_each(
                |(merge, val)|
                {
                    *merge = val + shift;
                }
            );


    }
    let shift = norm_ln_prob(&mut merged_log_prob);
    aligned_intervals.par_iter_mut()
        .for_each(
            |interval|
            interval.iter_mut()
                .for_each(|val| *val += shift)
        );
    Ok(
        (e_hist, merged_log_prob, aligned_intervals)
    )
}

#[allow(clippy::type_complexity)]
pub(crate) fn align<Hist>(container: &[(&[f64], &Hist)]) -> Result<(Vec<usize>, Vec<usize>, Vec<Vec<f64>>, Hist), HistErrors>
where Hist: HistogramCombine + Send + Sync
{
    let hists: Vec<_> = container.iter()
        .map(|v| v.1)
        .collect();
    
    let e_hist = Hist::encapsulating_hist(&hists)?;

    let derivatives: Vec<_> = container.par_iter()
        .map(|v| derivative_merged(v.0))
        .collect();

    let alignment = hists.iter()
        .zip(hists.iter().skip(1))
        .map(|(&left, &right)| left.align(right))
        .collect::<Result<Vec<_>, _>>()?;

    let merge_points = calc_merge_points(&alignment, &derivatives);

    let log_prob = container.iter()
        .map(|v| v.0.into())
        .collect();

    Ok(
        (
            merge_points,
            alignment,
            log_prob,
            e_hist
        )
    )
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

// TODO maybe rename function?
fn calc_z(log10_vec: &[Vec<f64>], alignment: &[usize]) -> Result<Vec<f64>, GlueErrors>
{
    let mut z_vec = Vec::with_capacity(alignment.len());
    for (i, &align) in alignment.iter().enumerate()
    {
        let prob_right = &log10_vec[i+1];
        let prob_left = &log10_vec[i][align..];
        let overlap_size = prob_right.len().min(prob_left.len());
        
        let sum = prob_left.iter().zip(prob_right.iter())
            .fold(0.0, |acc, (&p, &c)| p - c + acc);
        let mut z = sum / overlap_size as f64;
        // also correct for adjustment of prev
        if let Some(val) = z_vec.last() {
            z += val;
        }
        z_vec.push(z);
    }
    Ok(z_vec)
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
    for i in 0..first_log.len() {
        glue_count[i] = 1;
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
                    *count += 1;
                    *glued = if glued.is_finite(){
                         *glued + prob
                    } else {
                        prob
                    };
                }
            );
    }

    glue_log_density.iter_mut()
        .zip(glue_count.iter())
        .for_each(|(log, &count)| {
            if count > 0 {
                *log /= count as f64;
            }
        });
    
    Ok(glue_log_density)
}

// TODO Rename function?
pub(crate) fn no_derive_merged_and_aligned<Hist>(
    alignment: Vec<usize>,
    mut log_prob: Vec<Vec<f64>>,
    e_hist: Hist
) -> ReplicaGlued<Hist>
where Hist: HistogramCombine + Histogram,
{
    if alignment.is_empty(){
        // entering this means we only have 1 interval!
        assert_eq!(log_prob.len(), 1);
        norm_log10_sum_to_1(&mut log_prob[0]);
        let glued = log_prob[0].clone();
        return ReplicaGlued{
            base: LogBase::Base10,
            encapsulating_histogram: e_hist,
            aligned: log_prob,
            glued,
            alignment
        };
    }

    // calc z
    let z_vec = calc_z(&log_prob, &alignment).expect("Unable to calculate Z in glueing");

    println!("{:?}", z_vec);

    // correct height
    height_correction(&mut log_prob, &z_vec);
    // renaming
    let mut aligned_intervals = log_prob;

    // glueing together
    let mut glued_log_density = glue_no_derive(e_hist.bin_count(), &aligned_intervals, &alignment)
        .expect("Glue error!");

    // now norm the result
    // TODO AM I REALLY BASE 10 here?
    norm_log10_sum_to_1(&mut glued_log_density);

    let shift = glued_log_density[0] - aligned_intervals[0][0];

    aligned_intervals
        .iter_mut()
        .flat_map(|vec| vec.iter_mut())
        .for_each(|v| *v += shift);

    ReplicaGlued{
        base: LogBase::Base10,
        encapsulating_histogram: e_hist,
        aligned: aligned_intervals,
        glued: glued_log_density,
        alignment
    }
}


pub(crate) fn ln_to_log10(slice: &mut [f64])
{
    slice.iter_mut()
            .for_each(|val| *val *= LOG10_E);
}

pub(crate) fn log10_to_ln(slice: &mut [f64])
{
    slice.iter_mut()
        .for_each(|val| *val /= LOG10_E);
}

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
    encapsulating_histogram: Hist,
    glued: Vec<f64>,
    aligned: Vec<Vec<f64>>,
    base: LogBase,
    alignment: Vec<usize>
}

impl<Hist> ReplicaGlued<Hist>
{
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

    pub fn base(&self) -> LogBase
    {
        self.base
    }

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
where T: histogram::HasUnsignedVersion + num_traits::PrimInt + std::fmt::Display,
    T::Unsigned: num_traits::Bounded + HasUnsignedVersion<LeBytes=T::LeBytes> 
    + num_traits::WrappingAdd + num_traits::ToPrimitive + std::ops::Sub<Output=T::Unsigned>
{
    pub fn write<W: std::io::Write>(&self, mut writer: W) -> std::io::Result<()>
    {
        writeln!(writer, "#bin merged interval0 …")?;
        writeln!(writer, "#Base {:?}", self.base)?;

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
} 