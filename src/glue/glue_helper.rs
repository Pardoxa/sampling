use crate::*;
use std::cmp::*;
use std::convert::*;


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

/// Glues together probabilities
/// size is original_hist.bin_count()
pub(crate) fn glue(size: usize, log10_vec: &Vec<Vec<f64>>, left_list: &Vec<usize>, right_list: &Vec<usize>) -> Result<Vec<f64>, GlueErrors>
{
    let mut glue_log_density = vec![f64::NAN; size];


    // init - first interval can be copied for better performance
    let first_log = match log10_vec.first(){
        Some(interval) => interval.as_slice(),
        None => return Err(GlueErrors::EmptyList)
    };
    let l = *left_list.first().unwrap();
    let r = *right_list.first().unwrap();
    if r >= glue_log_density.len() {
        return Err(GlueErrors::OutOfBounds);
    }
    glue_log_density[l..=r].copy_from_slice(first_log);
    let mut glue_count = vec![0_usize; glue_log_density.len()];
    for i in l..=r {
        glue_count[i] = 1;
    }

    for (i, log_vec) in log10_vec.iter().enumerate().skip(1)
    {
        let left = left_list[i];
        let right = right_list[i];
        glue_log_density[left..=right].iter_mut()
            .zip(glue_count[left..=right].iter_mut())
            .zip(log_vec.iter())
            .for_each(|((res, count), &val)| {
                *count += 1;
                if res.is_finite(){
                    *res += val;
                } else {
                    *res = val;
                }
            });
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

pub(crate) fn height_correction(log10_vec: &mut Vec<Vec<f64>>, z_vec: &Vec<f64>){
    log10_vec.iter_mut()
        .skip(1)
        .zip(z_vec.iter())
        .for_each( |(vec, &z)|
            vec.iter_mut()
                .for_each(|val| *val += z )
        );
}

pub(crate) fn calc_z(log10_vec: &Vec<Vec<f64>>, left_list: &Vec<usize>, right_list: &Vec<usize>) -> Result<Vec<f64>, GlueErrors>
{
    let mut z_vec = Vec::with_capacity(left_list.len() - 1);
    for i in 1..left_list.len()
    {
            let left_prev = left_list[i - 1];
            let left = left_list[i];
            let l_m = left.max(left_prev);
            let right_prev = right_list[i - 1];
            let right = right_list[i];
            let r_m = right.min(right_prev);
            if l_m >= r_m {
                return Err(GlueErrors::NoOverlap);
            }
            let overlap_size = r_m - l_m;
            let (prev, cur) = if left_prev >= left{
                let diff = left_prev - left;
                (
                    &log10_vec[i - 1][0..=overlap_size],
                    &log10_vec[i][diff..=diff+overlap_size]
                )
            } else {
                let diff = left - left_prev;
                (
                    &log10_vec[i - 1][diff..=diff+overlap_size],
                    &log10_vec[i][0..=overlap_size]
                )
            };
            let sum = prev.iter().zip(cur.iter())
                .fold(0.0, |acc, (&p, &c)| p - c + acc);
            let mut z = sum / prev.len() as f64;
            // also correct for adjustment of prev
            if let Some(val) = z_vec.last() {
                z += val;
            }
            z_vec.push(z);
    }
    Ok(z_vec)
}

pub(crate) fn get_index<T>(val: &T, borders: &Vec<T>) -> Result<usize, GlueErrors>
where 
    T: PartialOrd
{
    let mut error = false;
    let index = borders.binary_search_by(
        |probe|{
            probe.partial_cmp(val).unwrap_or_else(|| {
                    error = true;
                    Ordering::Equal
                }
            )
        }
    );
    if error {
        return Err(GlueErrors::BinarySearch);
    }
    index.map_err(|_| GlueErrors::BinarySearch)
}

/// calles subtract_max on all contained vectors
pub(crate) fn inner_subtract_max(log10_vec: &mut Vec<Vec<f64>>)
{
    log10_vec.iter_mut()
        .for_each(
        |v| 
        {
            subtract_max(v);
        }
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