
use std::{
    ops::RangeInclusive,
    borrow::Borrow
};
use paste::paste;
use super::{
    HistFastIterHelper,
    Binning,
    HasUnsignedVersion,
    to_u,
    HistErrors
};

/// Generic binning meant for any integer type
pub struct FastIntBinning<T>{
    /// left bin border, inclusive
    pub start: T,
    /// right bin border, inclusive
    pub end_inclusive: T
}

macro_rules! impl_binning {
    (
        $t:ty
    ) => {
        
        paste!{
            #[doc = "Efficient binning for `" $t "` with bins of width 1"]
            pub type [<FastBinning $t:upper>] = FastIntBinning<$t>;
        }

        impl paste!{[<FastBinning $t:upper>]}{
            /// # Create a new Binning
            /// * both borders are inclusive
            /// * each bin has width 1
            /// # Panics
            /// * if `start` is smaller than `end_inclusive`
            pub fn new(start: $t, end_inclusive: $t) -> Self{
                assert!(start <= end_inclusive);
                Self {start, end_inclusive }
            }

            /// Get left border, inclusive
            pub fn left(&self) -> $t {
                self.start
            }

            /// Get right border, inclusive
            pub fn right(&self) -> $t
            {
                self.end_inclusive
            }

            /// # Returns the range covered by the bins as a `RangeInclusive<T>`
            pub fn range_inclusive(&self) -> RangeInclusive<$t>
            {
                self.start..=self.end_inclusive
            }

            paste!{
                #[doc = "# Iterator over all the bins\
                \nSince the bins have width 1, a bin can be defined by its corresponding value \
                which we can iterate over.\n\
                # Example\n\
                ```\n\
                use sampling::histogram::" [<FastBinning $t:upper>] ";\n\
                let binning = " [<FastBinning $t:upper>] "::new(2,5);\n\
                let vec: Vec<_> = binning.bin_iter().collect();\n\
                assert_eq!(&vec, &[2, 3, 4, 5]);\n\
                ```"]
                pub fn bin_iter(&self) -> impl Iterator<Item=$t>
                {
                    HistFastIterHelper{
                        current: self.start,
                        right: self.end_inclusive,
                        invalid: false
                    }
                
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

 
        impl Binning<$t> for paste!{[<FastBinning $t:upper>]} {
            fn get_bin_len(&self) -> usize 
            {
                //let len = self.bins_m1();
                todo!()
            }

            fn get_bin_index<V: Borrow<$t>>(&self, _: V) -> Option<usize>{
                todo!()
            }

            /// Does a value correspond to a valid bin?
            fn is_inside<V: Borrow<$t>>(&self, _: V) -> bool{
                todo!()
            }

            /// Opposite of `is_inside`
            fn not_inside<V: Borrow<$t>>(&self, _: V) -> bool{
                todo!()
            }

            /// get the left most border (inclusive)
            fn first_border(&self) -> $t{
                todo!()
            }

            fn last_border(&self) -> $t{
                todo!()
            }

            #[inline(always)]
            fn last_border_is_inclusive(&self) -> bool
            {
                true
            }

            /// # calculates some sort of absolute distance to the nearest valid bin
            /// * any invalid numbers (like NAN or INFINITY) should have the highest distance possible
            /// * if a value corresponds to a valid bin, the distance should be zero
            fn distance<V: Borrow<$t>>(&self, _: V) -> f64{
                todo!{}
            }

            fn borders_clone(&self) -> Result<Vec<$t>, HistErrors>{
                todo!()
            }
        }
    };
    (
        $($t:ty),* $(,)?
    ) => {
        $(
            impl_binning!($t);
        )*
    }
}
impl_binning!(
    u8, 
    i8,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64,
    u128,
    i128
);