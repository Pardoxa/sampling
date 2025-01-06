use super::*;
use std::{
    marker::PhantomData,
    borrow::Borrow,
    sync::atomic::AtomicUsize
};

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

/// # Provides Histogram functionality
/// * Is automatically implemented for any type that implements Binning
#[derive(Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct GenericHist<B, T>{
    /// The binning
    pub(crate) binning: B,
    /// Here we count the hits of the histogram
    pub(crate) hits: Vec<usize>,
    /// type that is counted
    pub(crate) phantom: PhantomData<T>
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

    /// Get reference to underlying binning
    pub fn binning(&self) -> &B
    {
        &self.binning
    }

    /// Converts self into an atomic histogram,
    /// such that you may parallelize your operations
    pub fn into_atomic_generic_hist(self) -> AtomicGenericHist<B, T>
    {
        self.into()
    }
}

impl<B, T> Histogram for GenericHist<B, T>
where B: Binning<T> {
    fn hist(&self) -> &Vec<usize> {
        &self.hits
    }

    #[inline]
    fn any_bin_zero(&self) -> bool {
        self.hits.iter().any(|h| *h == 0)
    }

    fn bin_count(&self) -> usize {
        debug_assert_eq!(self.binning.get_bin_len(), self.hits.len());
        self.hits.len()
    }

    #[inline]
    fn increment_index(&mut self, index: usize) -> Result<(), HistErrors> {
        let entry = self.hits
            .get_mut(index)
            .ok_or(HistErrors::OutsideHist)?;
        *entry += 1;
        Ok(())
    }

    #[inline]
    fn increment_index_by(&mut self, index: usize, count: usize) -> Result<(), HistErrors> {
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

impl<T,B> HistogramVal<T> for GenericHist<B, T>
where B: Binning<T>
{
    #[inline(always)]
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

    #[inline(always)]
    fn count_val<V: Borrow<T>>(&mut self, val: V) -> Result<usize, HistErrors> {
        let index = self.get_bin_index(val)?;
        self.increment_index(index)
            .map(|_| index)
    }

    fn not_inside<V: Borrow<T>>(&self, val: V) -> bool {
        self.binning.not_inside(val)
    }

    fn last_border(&self) -> T {
        self.binning.last_border()
    }

    #[inline]
    fn last_border_is_inclusive(&self) -> bool {
        self.binning.last_border_is_inclusive()
    }

    fn distance<V: Borrow<T>>(&self, val: V) -> f64 {
        self.binning.distance(val)
    }

    fn bin_enum_iter(&self) -> Box<dyn Iterator<Item=Bin<T>>> {
        self.binning().bin_iter()
    }
}

impl<B, T> From<B> for GenericHist<B, T>
where B: Binning<T>
{
    fn from(binning: B) -> Self {
        GenericHist::new(binning)
    }
}

impl<B, T> From<AtomicGenericHist<B, T>> for GenericHist<B, T>
{
    fn from(generic_atomic: AtomicGenericHist<B, T>) -> Self {
        let hits = generic_atomic.hits
            .into_iter()
            .map(AtomicUsize::into_inner)
            .collect();

        Self { 
            binning: generic_atomic.binning, 
            hits, 
            phantom: generic_atomic.phantom 
        }
    }
}