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