use super::*;
use std::{
    marker::PhantomData,
    borrow::Borrow,
    sync::atomic::AtomicUsize
};


/// # Provides Histogram functionallity
/// * Is automatically implemented for any type that implements Binning
pub struct GenericAtomicHist<B, T>{
    /// The binning
    binning: B,
    /// Here we count the hits of the histogram
    hits: Vec<AtomicUsize>,
    /// type that is counted
    phantom: PhantomData<T>
}

impl<B, T> GenericAtomicHist<B, T> 
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
}

impl<B, T> AtomicHistogram for GenericAtomicHist<B, T>
{
    fn hist(&self) -> &[AtomicUsize] {
        &self.hits
    }

    fn count_multiple_index(&self, index: usize, count: usize) -> Result<(), HistErrors> {
        let element = self.hits
            .get(index)
            .ok_or(HistErrors::OutsideHist)?;
        element.fetch_add(count, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn reset(&mut self) {
        self.hits
            .iter_mut()
            .for_each(|val| val.store(0, std::sync::atomic::Ordering::SeqCst));
    }
}

impl<B, T> AtomicHistogramVal<T> for GenericAtomicHist<B, T>
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

    fn borders_clone(&self) -> Result<Vec<T>, HistErrors> {
        // remove this function
        unimplemented!()
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