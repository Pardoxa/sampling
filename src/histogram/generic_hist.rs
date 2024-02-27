use super::*;
use std::{
    marker::PhantomData,
    borrow::Borrow
};


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

    /// Get reference to underlying binning
    pub fn binning(&self) -> &B
    {
        &self.binning
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
    fn count_index(&mut self, index: usize) -> Result<(), HistErrors> {
        let entry = self.hits
            .get_mut(index)
            .ok_or(HistErrors::OutsideHist)?;
        *entry += 1;
        Ok(())
    }

    #[inline]
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
        self.count_index(index)
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

