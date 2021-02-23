use crate::glue_helper::*;
use crate::*;
use rayon::prelude::*;

pub(crate) fn norm_ln_prob(ln_prob: &mut[f64]) -> f64
{
    subtract_max(ln_prob);
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

    shift

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

    (merged_log_prob, e_hist)
}

pub(crate) fn merged_and_aligned<'a, Hist: 'a, I>(
    hists: I,
    merge_points: Vec<usize>,
    alignment: Vec<usize>,
    log_prob: Vec<Vec<f64>>,
    e_hist: Hist
) -> Result<(Hist, Vec<f64>, Vec<Vec<f64>>), HistErrors>
where Hist: HistogramCombine + Histogram,
    I: Iterator<Item = &'a Hist>
{
    let mut merged_log_prob = vec![f64::NAN; e_hist.bin_count()];

    let mut aligned_intervals = vec![merged_log_prob.clone(); alignment.len() + 1];

    aligned_intervals[0][..log_prob[0].len()]
        .copy_from_slice(&log_prob[0]);
    
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
    Ok(
        (e_hist, merged_log_prob, aligned_intervals)
    )
}

pub(crate) fn align<Hist>(container: &Vec<(&[f64], &Hist)>) -> Result<(Vec<usize>, Vec<usize>, Vec<Vec<f64>>, Hist), HistErrors>
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

    let log_prob = container.into_iter()
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
