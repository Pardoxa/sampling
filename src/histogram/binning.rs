use super::*;
use std::borrow::Borrow;
/// # Implements Binning
/// * Part of a histogram, but without the capability of counting stuff
/// 
/// # Note
/// * Currently binning is not used in this lib. But I plan to Refactor the histograms 
/// in the next breaking release such that one only has to implement Binning 
/// and can create a histogram from that
pub trait Binning<T>{
    /// convert val to the respective binning index
    fn get_bin_index<V: Borrow<T>>(&self, val: V) -> Option<usize>;

    /// # Get the number of underlying bins
    /// * note: if more than usize::MAX bins are there, usize::MAX is returned
    fn get_bin_len(&self) -> usize;

    /// # binning borders
    /// * the borders used to bin the values
    /// * any val which fulfills `self.border[i] <= val < self.border[i + 1]` 
    /// will get index `i`.
    /// * **Note** that the last border is usually exclusive
    fn borders_clone(&self) -> Result<Vec<T>, HistErrors>;

    /// Does a value correspond to a valid bin?
    fn is_inside<V: Borrow<T>>(&self, val: V) -> bool;

    /// Opposite of `is_inside`
    fn not_inside<V: Borrow<T>>(&self, val: V) -> bool;

    /// get the left most border (inclusive)
    fn first_border(&self) -> T;

    /// * get last border from the right
    /// * Note: this border might be inclusive or exclusive
    fn second_last_border(&self) -> T;

    /// # calculates some sort of absolute distance to the nearest valid bin
    /// * any invalid numbers (like NAN or INFINITY) should have the highest distance possible
    /// * if a value corresponds to a valid bin, the distance should be zero
    fn distance<V: Borrow<T>>(&self, val: V) -> f64;
}

mod binning_int_fast;
pub use binning_int_fast::*;