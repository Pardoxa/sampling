use super::*;
use std::borrow::Borrow;

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};



/// # Definition of a Bin
/// * Note: Most (currently all) implementations use more efficient representations of the bins underneath,
/// but are capable of returning the bins in this representation on request
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum Bin<T>
{
    /// The bin consists of a single value. A value is inside the bin if it equals this value
    SingleValued(T),
    /// The bin is defined by two inclusive borders (left, right).
    /// a value is inside the bin, if left <= value <= right
    InclusiveInclusive(T, T),
    /// The bin is defined by an inclusive and an exclusive border (left, right).
    /// a value is inside the bin, if left <= value < right
    InclusiveExclusive(T, T),
    /// The bin is defined by an exclusive and an inclusive border (left, right).
    /// a value is inside the bin, if left < value <= right
    ExclusiveInclusive(T, T),
    /// The bin is defined by two exclusive borders (left, right).
    /// a value is inside the bin, if left < value < right
    ExclusiveExclusive(T, T)
}


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

    /// # Iterates over all bins
    /// * Note: Most (currently all) implementations use more efficient representations of the bins underneath,
    /// but are capable of returning the bins in this representation on request
    /// * Also look if the Binning you use implements another method for the bin borders, e.g., `single_valued_bin_iter`,
    /// as that is more efficient
    fn bin_iter(&self) -> Box<dyn Iterator<Item=Bin<T>>>;

    /// Does a value correspond to a valid bin?
    fn is_inside<V: Borrow<T>>(&self, val: V) -> bool;

    /// Opposite of `is_inside`
    fn not_inside<V: Borrow<T>>(&self, val: V) -> bool;

    /// get the left most border (inclusive)
    fn first_border(&self) -> T;

    /// * get last border from the right
    /// * Note: this border might be inclusive or exclusive
    fn last_border(&self) -> T;

    /// # True if last border is inclusive, false otherwise
    /// * For most usecases this will return a constant value,
    /// as this is likely only dependent on the underlying type and not 
    /// on something that changes dynamically
    fn last_border_is_inclusive(&self) -> bool;

    /// # calculates some sort of absolute distance to the nearest valid bin
    /// * any invalid numbers (like NAN or INFINITY) should have the highest distance possible
    /// * if a value corresponds to a valid bin, the distance should be zero
    fn distance<V: Borrow<T>>(&self, val: V) -> f64;
}

mod binning_int_fast;
pub use binning_int_fast::*;