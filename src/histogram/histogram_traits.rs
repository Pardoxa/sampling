use std::{borrow::*,num::NonZeroUsize, cmp::Ordering, marker::PhantomData};

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

/// # Implements Binning
/// * Part of a histogram, but without the capability of counting stuff
/// 
/// # Note
/// * Currently binning is not used in this lib. But I plan to Refactor the histograms 
/// in the next breaking release such that one only has to implement Binning 
/// and can create a histogram from that
pub trait Binning<T>{
    /// convert val to the respective binning indes
    fn get_bin_index<V: Borrow<T>>(&self, val: V) -> Option<usize>;

    /// Get the number of underlying bins
    fn get_bin_len(&self) -> usize;

    /// # binning borders
    /// * the borders used to bin the values
    /// * any val which fullfills `self.border[i] <= val < self.border[i + 1]` 
    /// will get index `i`.
    /// * **Note** that the last border is exclusive
    /// TODO Think about this one
    fn borders_clone(&self) -> Result<Vec<T>, HistErrors>;

    /// Does a value correspond to a valid bin?
    fn is_inside<V: Borrow<T>>(&self, val: V) -> bool;

    /// Opposite of `is_inside`
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


/// # Provides Histogram functionallity
/// * Is automatically implemented for any type that implements Binning
pub struct GenericHist<B, T>{
    /// The binning
    binning: B,
    /// Here we count the hits of the histogram
    hits: Vec<usize>,
    /// type that is counted
    phantom: PhantomData<T>
}

impl<B, T> GenericHist<B, T> 
where B: Binning<T>{
    /// Create a new histogram from an arbitrary binning
    pub fn new(binning: B) -> Self{
        Self{
            hits: vec![0; binning.get_bin_len()],
            binning ,
            phantom: PhantomData
        }
    }
}

impl<B, T> Histogram for GenericHist<B, T>
where B: Binning<T> {
    fn hist(&self) -> &Vec<usize> {
        &self.hits
    }

    fn any_bin_zero(&self) -> bool {
        self.hits.iter().any(|h| *h == 0)
    }

    fn bin_count(&self) -> usize {
        debug_assert_eq!(self.binning.get_bin_len(), self.hits.len());
        self.hits.len()
    }

    fn count_index(&mut self, index: usize) -> Result<(), HistErrors> {
        let entry = self.hits
            .get_mut(index)
            .ok_or(HistErrors::OutsideHist)?;
        *entry += 1;
        Ok(())
    }

    fn count_multiple_index(&mut self, index: usize, count: usize) -> Result<(), HistErrors> {
        let entry = self.hits
            .get_mut(index)
            .ok_or(HistErrors::OutsideHist)?;
        *entry += count;
        Ok(())
    }

    fn reset(&mut self) {
        self.hits.iter_mut().for_each(|val| *val = 0);
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

impl<T,B> HistogramVal<T> for GenericHist<B, T>
where B: Binning<T>
{
    fn get_bin_index<V: Borrow<T>>(&self, val: V) -> Result<usize, HistErrors> {
        self.binning.get_bin_index(val)
            .ok_or(HistErrors::OutsideHist)
    }

    fn first_border(&self) -> T {
        self.binning.first_border()
    }

    fn is_inside<V: Borrow<T>>(&self, val: V) -> bool {
        self.binning.is_inside(val)
    }

    fn count_val<V: Borrow<T>>(&mut self, val: V) -> Result<usize, HistErrors> {
        let index = self.get_bin_index(val)?;
        self.count_index(index)
            .map(|_| index)
    }

    fn not_inside<V: Borrow<T>>(&self, val: V) -> bool {
        self.binning.not_inside(val)
    }

    fn second_last_border(&self) -> T {
        self.binning.second_last_border()
    }

    fn distance<V: Borrow<T>>(&self, val: V) -> f64 {
        self.binning.distance(val)
    }

    fn borders_clone(&self) -> Result<Vec<T>, HistErrors> {
        self.binning.borders_clone()
    }
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