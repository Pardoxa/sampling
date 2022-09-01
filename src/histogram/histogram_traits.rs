use std::{borrow::*,num::NonZeroUsize, cmp::Ordering};

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

/// # Implements histogram
/// * anything that implements `Histogram` should also implement the trait `HistogramVal`
pub trait Histogram {
    /// # `self.hist[index] += 1`, `Err()` if `index` out of bounds
    #[inline(always)]
    fn count_index(&mut self, index: usize) -> Result<(), HistErrors>{
        self.count_multiple_index(index, 1)
    }

    /// # `self.hist[index] += count`, `Err()` if `index` out of bounds
    fn count_multiple_index(&mut self, index: usize, count: usize) -> Result<(), HistErrors>;

    /// # the created histogram
    fn hist(&self) -> &Vec<usize>;
    /// # How many bins the histogram contains
    #[inline(always)]
    fn bin_count(&self) -> usize
    {
        self.hist().len()
    }
    /// reset the histogram to zero
    fn reset(&mut self);

    /// check if any bin was not hit yet
    fn any_bin_zero(&self) -> bool
    {
        self.hist()
            .iter()
            .any(|&val| val == 0)
    }
}

/// * trait used for mapping values of arbitrary type `T` to bins
/// * used to create a histogram
pub trait HistogramVal<T>{
    /// convert val to the respective histogram index
    fn get_bin_index<V: Borrow<T>>(&self, val: V) -> Result<usize, HistErrors>;
    /// count val. `Ok(index)`, if inside of hist, `Err(_)` if val is invalid
    fn count_val<V: Borrow<T>>(&mut self, val: V) -> Result<usize, HistErrors>;
    /// # binning borders
    /// * the borders used to bin the values
    /// * any val which fullfills `self.border[i] <= val < self.border[i + 1]` 
    /// will get index `i`.
    /// * **Note** that the last border is exclusive
    fn borders_clone(&self) -> Result<Vec<T>, HistErrors>;
    /// does a value correspond to a valid bin?
    fn is_inside<V: Borrow<T>>(&self, val: V) -> bool;
    /// opposite of `is_inside`
    fn not_inside<V: Borrow<T>>(&self, val: V) -> bool;
    /// get the left most border (inclusive)
    fn first_border(&self) -> T;

    /// * get second last border from the right
    /// * should be the same as `let b = self.borders_clone().expect("overflow"); assert_eq!(self.second_last_border(), b[b.len()-2])`
    fn second_last_border(&self) -> T;
    /// # calculates some sort of absolute distance to the nearest valid bin
    /// * any invalid numbers (like NAN or INFINITY) should have the highest distance possible
    /// * if a value corresponds to a valid bin, the distance should be zero
    fn distance<V: Borrow<T>>(&self, val: V) -> f64;
}

/// Distance metric for how far a value is from a valid interval
pub trait HistogramIntervalDistance<T> {
    /// # Distance metric for how far a value is from a valid interval
    /// * partitions in more intervals, checks which bin interval a bin corresponds to 
    /// and returns distance of said interval to the target interval
    /// * used for heuristiks
    /// * overlap should be bigger 0, otherwise it will be set to 1
    fn interval_distance_overlap<V: Borrow<T>>(&self, val: V, overlap: NonZeroUsize) -> usize;
}



/// # Your Interval is to large to sample in a resonable amound of time? No problem
/// In WangLandau or EntropicSampling, you can split your interval
/// in smaller, overlapping intervals and "glue" them together later on
pub trait HistogramPartition: Sized
{
    /// # partition the interval
    /// * returns Vector of `n` histograms, that together 
    /// ## parameter
    /// * `n` number of resulting intervals
    /// * `overlap` How much overlap should there be?
    /// ## To understand `overlap`, we have to look at the formula for the i_th interval in the result vector:
    /// let `left` be the left border of `self` and `right` be the right border of self
    /// * left border of interval i = left + i * (right - left) / (n + overlap)
    /// * right border of interval i = left + (i + overlap) * (right - left) / (n + overlap)
    fn overlapping_partition(&self, n: usize, overlap: usize) -> Result<Vec<Self>, HistErrors>;
}

/// # Used to get a histogram, which contains the smaller histograms
pub trait HistogramCombine: Sized
{
    /// # Create a histogram, which encapsulates the histograms passed
    /// # possible errors
    /// * bin size of histograms is unequal
    /// * bins do not align
    fn encapsulating_hist<S>(hists: &[S]) -> Result<Self, HistErrors>
    where S: Borrow<Self>;

    /// # Get bin difference between histograms
    /// * index of bin of self corresponding to the leftest bin of `right`
    fn align<S>(&self, right: S)-> Result<usize, HistErrors>
    where S: Borrow<Self>;
}

/// Trait for comparing two intervals
pub trait IntervalOrder
{
    /// Will compare leftest bin first.
    /// if they are equal: will compare right bin
    fn left_compare(&self, other: &Self) -> Ordering;
}

/// Possible Errors of the traits `Histogram` and `HistogramVal`
#[derive(Debug, Clone, Eq, PartialEq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum HistErrors{
    /// A histogram without any bins does not make sense!
    NoBins,

    /// Nothing can hit the bin! (left >= right?)
    IntervalWidthZero,

    /// Invalid value
    OutsideHist,

    /// Underflow occured
    Underflow,

    /// Overflow occured,
    Overflow,

    /// Error while casting to usize
    UsizeCastError,
    
    /// Something went wrong wile casting!
    CastError,

    /// Could be NAN, INFINITY or similar
    InvalidVal,

    /// Cannot create requested interval with 
    /// bins, that all have the same width!
    ModuloError,

    /// Unable to perform operation on empty slice
    EmptySlice
}