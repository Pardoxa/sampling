use super::*;
use std::{
    marker::PhantomData,
    borrow::Borrow,
    sync::atomic::{
        AtomicUsize,
        Ordering
    }
};

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};


/// # Provides Histogram functionality
/// * Is automatically implemented for any type that implements Binning
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct AtomicGenericHist<B, T>{
    /// The binning
    pub(crate) binning: B,
    /// Here we count the hits of the histogram
    pub(crate) hits: Vec<AtomicUsize>,
    /// type that is counted
    pub(crate) phantom: PhantomData<T>
}

impl<B, T> AtomicGenericHist<B, T> 
where B: Binning<T>{
    /// Create a new histogram from an arbitrary binning
    pub fn new(binning: B) -> Self{
        let hits = (0..binning.get_bin_len())
            .map(|_| AtomicUsize::new(0))
            .collect();
        Self{
            hits,
            binning ,
            phantom: PhantomData
        }
    }

    /// Get reference to internal binning
    pub fn binning(&self) -> &B
    {
        &self.binning
    }

    /// # Total number of hits
    /// Iterates through [Self::hist] and calculates the total hits of the histogram.
    /// Saturates at usize::MAX
    pub fn total_hits(&self) -> usize
    {
        let mut total: usize = 0;
        self.hist()
            .iter()
            .for_each(
                 |val|
                 {
                    total = total.saturating_add(
                        val.load(Ordering::Relaxed)
                    )
                 }
            );
        total
    }

    /// # Iterator over hit count of bins
    /// Note: You can get an iterator over the corresponding bins via the underlying binning,
    /// i.e., [Self::binning]
    /// 
    /// Also note: Since Atomic hist allows you to iterate in parallel, you are also able 
    /// to call this iterator, while another thread is still incrementing the histogram.
    /// The output of the iterator will be valid, but depending on the timings it may or may not 
    /// include (parts of) the hits from the thread running in parallel
    pub fn hits_iter(&'_ self) -> impl Iterator<Item=usize> + '_
    {
        self.hits
            .iter()
            .map(
                |val| val.load(Ordering::Relaxed)
            )
    }

    /// Converts self into a Generic hist, i.e., a histogram without the atomic part.
    /// This can be easier to deal with when you are single threaded and might have more functionality
    pub fn into_generic_hist(self) -> GenericHist<B, T>
    {
        self.into()
    }
}


impl<B, T> AtomicHistogram for AtomicGenericHist<B, T>
{
    fn hist(&self) -> &[AtomicUsize] {
        &self.hits
    }

    fn count_multiple_index(&self, index: usize, count: usize) -> Result<(), HistErrors> {
        let element = self.hits
            .get(index)
            .ok_or(HistErrors::OutsideHist)?;
        element.fetch_add(count, Ordering::Relaxed);
        Ok(())
    }

    fn reset(&mut self) {
        self.hits
            .iter_mut()
            .for_each(|val| val.store(0, std::sync::atomic::Ordering::SeqCst));
    }
}

impl<B, T> AtomicHistogramVal<T> for AtomicGenericHist<B, T>
where B: Binning<T>

{
    fn get_bin_index<V: Borrow<T>>(&self, val: V) -> Result<usize, HistErrors> {
        let index = self.binning.get_bin_index(val)
            .ok_or(HistErrors::OutsideHist)?;
        self.count_index(index)
            .map(|_| index)
    }

    fn count_val<V: Borrow<T>>(&self, val: V) -> Result<usize, HistErrors> {
        let index = self.binning
            .get_bin_index(val)
            .ok_or(HistErrors::OutsideHist)?;
        self.count_index(index)
            .map(|_| index)
    }

    fn distance<V: Borrow<T>>(&self, val: V) -> f64 {
        self.binning.distance(val)
    }

    fn first_border(&self) -> T {
        self.binning.first_border()
    }

    fn is_inside<V: Borrow<T>>(&self, val: V) -> bool {
        self.binning.is_inside(val)
    }

    fn not_inside<V: Borrow<T>>(&self, val: V) -> bool {
        self.binning.not_inside(val)
    }

    fn last_border(&self) -> T {
        self.binning.last_border()
    }
}

impl<B, T> From<B> for AtomicGenericHist<B, T>
where B: Binning<T>
{
    fn from(binning: B) -> Self {
        AtomicGenericHist::new(binning)
    }
}

impl<B, T> From<GenericHist<B, T>> for AtomicGenericHist<B, T>
{
    fn from(generic: GenericHist<B, T>) -> Self {
        let hits = generic.hits
            .into_iter()
            .map(AtomicUsize::new)
            .collect();

        AtomicGenericHist{
            binning: generic.binning,
            phantom: generic.phantom,
            hits
        }
    }
}