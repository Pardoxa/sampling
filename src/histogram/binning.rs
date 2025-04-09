use super::*;
use std::borrow::Borrow;

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};


mod binning_int_fast;
pub use binning_int_fast::*;
mod binning_int_multi;
pub use binning_int_multi::*;

/// # Definition of a Bin
/// * Note: Most (currently all) implementations use more efficient representations of the bins underneath,
///   but are capable of returning the bins in this representation on request
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
/// Part of a histogram, but without the capability of counting stuff, i.e.,
/// you can use this to iterate through the bins or to get the bin index that a certain value would 
/// correspond to, but it does not contain counters to track this.
/// 
/// You can use this to create histograms, see [GenericHist] or [AtomicGenericHist]
pub trait Binning<T>{
    /// convert val to the respective binning index
    fn get_bin_index<V: Borrow<T>>(&self, val: V) -> Option<usize>;

    /// # Get the number of underlying bins
    /// * note: if more than usize::MAX bins are there, usize::MAX is returned
    fn get_bin_len(&self) -> usize;

    /// # Iterates over all bins
    /// Note: Most (currently all) implementations use more efficient representations of the bins underneath,
    ///     but are capable of returning the bins in this representation on request. 
    ///     So it's better to use the native iterator instead, if you can.
    fn bin_iter(&self) -> Box<dyn Iterator<Item=Bin<T>>>;

    /// Does a value correspond to a valid bin?
    fn is_inside<V: Borrow<T>>(&self, val: V) -> bool;

    /// Opposite of `is_inside`, so: is the value outside the valid range?
    fn not_inside<V: Borrow<T>>(&self, val: V) -> bool;

    /// get the left most border (inclusive). This is the first value that lies inside the binning
    fn first_border(&self) -> T;

    /// * If the last border is inclusive, this returns the largest value that is still inside the binning.
    /// * If the last border is exclusive, this is the first value which is **not** inside the binning.
    fn last_border(&self) -> T;

    /// # True if last border is inclusive, false otherwise
    /// For most use cases this will return a constant value,
    /// as this is likely only dependent on the underlying type and not 
    /// on something that changes dynamically
    fn last_border_is_inclusive(&self) -> bool;

    /// # calculates some sort of absolute distance to the nearest valid bin
    /// * any invalid numbers (like NAN or INFINITY) should have the highest distance possible
    /// * if a value corresponds to a valid bin, the distance should be zero
    fn distance<V: Borrow<T>>(&self, val: V) -> f64;

    /// # Convert binning into [GenericHist]
    /// Useful histogram for single threaded context's. 
    /// Otherwise [AtomicGenericHist] might be more useful (see also [Binning::to_generic_atomic_hist])
    fn to_generic_hist(self) -> GenericHist<Self, T>
    where Self: Sized
    {
        GenericHist::new(self)
    }

    /// # Convert binning into a [AtomicGenericHist]
    /// Useful histogram if you want to create the histogram in parallel, but otherwise has less functionality 
    /// than [GenericHist] (see also [Binning::to_generic_hist])
    fn to_generic_atomic_hist(self) -> AtomicGenericHist<Self, T>
    where Self: Sized
    {
        AtomicGenericHist::new(self)
    }
}

/// # Trait used to display bins
/// * This is, e.g., used by the glue writers to write the bins of the merged results
pub trait BinDisplay {
    /// What type does the displayable BinEntry have?
    type BinEntry;

    /// # Iterator over all the bins
    /// * you might require to use this if you are working with generics
    /// * if you are working with a specific type there is usually a more efficient implementation
    ///   that did not require the usage of dynamic traits (`dyn`) and that are thus more efficient,
    ///   consider using those instead
    fn display_bin_iter(&'_ self) -> Box<dyn Iterator<Item=Self::BinEntry> + '_>;

    /// # For writing a bin
    /// * How to write a bin to a file? If a bin consists, e.g., of an exclusive and inclusive border this 
    ///   might require the writing of two values. It could also be a single value instead.
    ///   It should be something the user expects from your binning, see write header
    fn write_bin<W: std::io::Write>(entry: &Self::BinEntry, writer: W) -> std::io::Result<()>;

    /// # Writing the header of the bin
    /// * This is intended to name, e.g., a column in a file. Output could be "SingleBin" or "BinBorderExclusive BinBorderInclusive"
    ///   and so on
    fn write_header<W: std::io::Write>(&self, writer: W) -> std::io::Result<()>;
}