use std::{
    ops::RangeInclusive,
    borrow::Borrow, fmt::Debug
};
use paste::paste;
use super::{
    Binning,
    HasUnsignedVersion,
    to_u,
    Bin,
    BinModIterHelper
};

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};


/// Generic binning meant for any integer type
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct BinningWithWidth<T>
where T: HasUnsignedVersion
{
    /// left bin border, inclusive
    start: T,
    /// right bin border, inclusive
    end_inclusive: T,
    /// how many numbers are in one bin?
    bin_width: T
}

macro_rules! other_binning {
    (
        $t:ty
    ) => {
        
        paste!{
            #[doc = "Efficient binning for `" $t "` with bins of width 1"]
            pub type [<Binning $t:upper>] = BinningWithWidth<$t>;
        }
        
        impl paste!{[<Binning $t:upper>]}{
            /// # Create a new Binning
            /// * both borders are inclusive
            /// * each bin has width 1
            /// # Panics
            /// * if `start` is smaller than `end_inclusive`
            /// * if bin_width <= 0
            #[inline(always)]
            pub fn new_inclusive(start: $t, end_inclusive: $t, bin_width: $t) -> Result<Self, <$t as HasUnsignedVersion>::Unsigned>{
                assert!(start <= end_inclusive);
                assert!(bin_width > 0);
                //
                let this = Self{
                    start, 
                    end_inclusive,
                    bin_width
                };
                let u_width = bin_width as <$t as HasUnsignedVersion>::Unsigned;
                let res = (this.bins_m1() % u_width) + 1;
                if res != u_width{
                    Err(res)
                } else {
                    Ok(this)
                }
            }

            /// Get left border, inclusive
            #[inline(always)]
            pub const fn left(&self) -> $t {
                self.start
            }

            /// Get right border, inclusive
            #[inline(always)]
            pub const fn right(&self) -> $t
            {
                self.end_inclusive
            }

            /// # Returns the range covered by the bins as a `RangeInclusive<T>`
            #[inline(always)]
            pub const fn range_inclusive(&self) -> RangeInclusive<$t>
            {
                self.start..=self.end_inclusive
            }

            paste!{
                #[doc = "# Iterator over all the bins\
                \nSince the bins have width 1, a bin can be defined by its corresponding value \
                which we can iterate over.\n\
                # Example\n\
                ```\n\
                use sampling::histogram::" [<Binning $t:upper>] ";\n\
                let binning = " [<Binning $t:upper>] "::new_inclusive(2,7,2).unwrap();\n\
                let vec: Vec<_> = binning.multi_valued_bin_iter().collect();\n\
                assert_eq!(&vec, &[(2, 3), (4, 5), (6, 7)]);\n\
                ```"]
                pub fn multi_valued_bin_iter(&self) -> impl Iterator<Item=($t, $t)>
                {
                    let width = self.bin_width as $t;
                    BinModIterHelper::new_unchecked(self.start, self.end_inclusive, width)
                }
            }

            /// # The amount of bins -1
            /// * minus 1 because if the bins are going over the entire range of the type,
            /// then I cannot represent the number of bins as this type
            /// 
            /// # Example
            /// If we look at an u8 and the range from 0 to 255, then this is 256 bins, which 
            /// cannot be represented as u8. To combat this, I return bins - 1.
            /// This works, because we always have at least 1 bin
            pub fn bins_m1(&self) -> <$t as HasUnsignedVersion>::Unsigned{
                let left = to_u(self.start);
                let right = to_u(self.end_inclusive);

                right - left
            }
        }

 
        impl Binning<$t> for paste!{[<Binning $t:upper>]} {
            #[inline(always)]
            fn get_bin_len(&self) -> usize 
            {
                (self.bins_m1() as usize).saturating_add(1)
            }

            /// # Get the respective bin index
            /// * Note: Obviously this breaks when the bin index cannot be represented as 
            /// `usize`
            #[inline(always)]
            fn get_bin_index<V: Borrow<$t>>(&self, val: V) -> Option<usize>{
                let val = *val.borrow();
                if self.is_inside(val)
                {
                    let bin_width = self.bin_width as <$t as HasUnsignedVersion>::Unsigned;
                    let index = (to_u(val) - to_u(self.start)) / bin_width;
                    Some(index as usize)
                } else{
                    None
                }
            }

            /// Does a value correspond to a valid bin?
            #[inline(always)]
            fn is_inside<V: Borrow<$t>>(&self, val: V) -> bool{
                (self.start..=self.end_inclusive).contains(val.borrow())
            }

            /// # Opposite of `is_inside`
            /// * I could also have called this `is_outside`, but I didn't
            #[inline(always)]
            fn not_inside<V: Borrow<$t>>(&self, val: V) -> bool{
                !self.is_inside(val)
            }

            /// get the left most border (inclusive)
            fn first_border(&self) -> $t{
                self.start
            }

            fn last_border(&self) -> $t{
                self.end_inclusive
            }

            #[inline(always)]
            fn last_border_is_inclusive(&self) -> bool
            {
                true
            }

            /// # calculates some sort of absolute distance to the nearest valid bin
            /// * if a value corresponds to a valid bin, the distance is zero
            fn distance<V: Borrow<$t>>(&self, v: V) -> f64{
                let val = v.borrow();
                if self.is_inside(val){
                    0.0
                } else {
                    let dist = if *val < self.start {
                        to_u(self.start) - to_u(*val)
                    } else {
                        // TODO Unit tests have to check if this is correct,
                        // before it was  val.saturating_sub(self.end_inclusive)
                        // but I think this here is better
                        to_u(*val) - to_u(self.end_inclusive)
                    };
                    dist as f64
                }
            }

            /// # Iterates over all bins
            /// * Note: This implementation use more efficient representations of the bins underneath,
            /// but are capable of returning the bins in this representation on request
            /// * Note also that this `Binning`  implements another method for the bin borders, i.e., `single_valued_bin_iter`.
            /// Consider using that instead, as it is more efficient
            fn bin_iter(&self) -> Box<dyn Iterator<Item=Bin<$t>>>{
                todo!() 
            }
        }
    };
    (
        $($t:ty),* $(,)?
    ) => {
        $(
            other_binning!($t);
        )*
    }
}

other_binning!(
    u8, 
    i8,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64,
    u128,
    i128,
    usize,
    isize
);