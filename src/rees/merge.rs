use{
    crate::{
        *,
        glue::derivative::*
    },
    rayon::prelude::*,
};

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
    
    // Nothing to align, only one interval here
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







