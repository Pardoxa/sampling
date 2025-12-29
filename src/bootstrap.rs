//! Bootstrap resampling functions
use rand::{seq::*, Rng};

/// returns reduced value + variance (estimated error is sqrt of variance)
/// Note, that you can use [bootstrap_copyable]
/// if your `N1` implements Copy
pub fn bootstrap<F, R, N1>(mut rng: R, samples: usize, data: &[N1], reduction: F) -> (f64, f64)
where
    F: Fn(&[&N1]) -> f64,
    R: Rng,
{
    let mut bootstrap_sample = Vec::with_capacity(data.len());

    let mut sum_sq = 0.0;
    let mut sum = 0.0;

    (0..samples).for_each(|_| {
        bootstrap_sample.clear();
        bootstrap_sample.extend((0..data.len()).map(|_| data.choose(&mut rng).unwrap()));
        let value = reduction(&bootstrap_sample);
        sum += value;
        sum_sq += value * value;
    });

    let factor = (samples as f64).recip();
    let mean = sum * factor;
    let variance = sum_sq * factor - mean * mean;
    (mean, variance)
}

/// Similar to [bootstrap] but for stuff that implements `Copy`. Likely more efficient in these cases
/// returns reduced value + variance (estimated error is sqrt of variance)
pub fn bootstrap_copyable<F, R, N1>(
    mut rng: R,
    samples: usize,
    data: &[N1],
    reduction: F,
) -> (f64, f64)
where
    F: Fn(&mut [N1]) -> f64,
    R: Rng,
    N1: Copy,
{
    let mut bootstrap_sample = Vec::with_capacity(data.len());

    let mut sum_sq = 0.0;
    let mut sum = 0.0;

    (0..samples).for_each(|_| {
        bootstrap_sample.clear();
        bootstrap_sample.extend((0..data.len()).map(|_| data.choose(&mut rng).unwrap()));
        let value = reduction(&mut bootstrap_sample);
        sum += value;
        sum_sq += value * value;
    });
    let factor = (samples as f64).recip();
    let mean = sum * factor;
    let variance = sum_sq * factor - mean * mean;
    (mean, variance)
}
