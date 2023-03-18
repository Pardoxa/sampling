use{
    crate::*,
    std::{convert::*, f64::consts::LOG10_E}
};

#[derive(Debug)]
/// Possible errors that can occur during gluing together
/// WangLandau intervals or Entropic Sampling intervals
pub enum GlueErrors{
    /// `original_hist.borders_clone()` failed
    BorderCreation(HistErrors),
    /// Nothing to be glued, glue interval list was empty
    EmptyList,
    /// Binary search failed - PartialOrd::partial_cmp returned None
    BinarySearch,
    /// # Glue interval and intervals to be glued do not match
    /// * Likely `original_hist` is to small
    OutOfBounds,
    /// The intervals need to overlap, otherwise no gluing can occur
    NoOverlap,
}

impl From<HistErrors> for GlueErrors{
    fn from(e: HistErrors) -> Self {
        GlueErrors::BorderCreation(e)
    }
}

/// # Normalize log10 probability density
/// * input: Slice containing log10 of (non normalized) probability density
/// * afterwards, it will be normalized, i.e., sum_i 10^log10_density\[i\] ≈ 1
pub fn norm_log10_sum_to_1(log10_density: &mut[f64]){
    // prevent errors due to small or very large numbers
    subtract_max(log10_density);

    // calculate actual sum in non log space
    let sum = log10_density.iter()
        .fold(0.0, |acc, &val| {
            if val.is_finite(){
               acc +  10_f64.powf(val)
            } else {
                acc
            }
        }  
    );
    
    let sum = sum.log10();
    log10_density.iter_mut()
        .for_each(|val| *val -= sum);
}



pub(crate) fn height_correction(log10_vec: &mut [Vec<f64>], z_vec: &[f64]){
    log10_vec.iter_mut()
        .skip(1)
        .zip(z_vec.iter())
        .for_each( |(vec, &z)|
            vec.iter_mut()
                .for_each(
                    |val| 
                    {
                        *val += z;
                    }
                )
        );
}





/// subtracts maximum, if it is finite
pub(crate) fn subtract_max(list: &mut[f64]) -> f64
{
    let max = list
        .iter()
        .copied()
        .fold(f64::NAN, f64::max);

    if max.is_finite() {
        list.iter_mut()
            .for_each(|val| *val -= max);
    }
    max
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