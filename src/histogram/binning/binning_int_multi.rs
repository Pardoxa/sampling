use std::{
    ops::RangeInclusive,
    borrow::Borrow, fmt::Debug
};
use paste::paste;
use crate::HistogramVal;

use super::{
    to_u, 
    Bin, 
    BinModIterHelper, 
    Binning, 
    GenericHist, 
    HasUnsignedVersion, 
    HistogramCombine, 
    HistErrors
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
            #[doc = "Efficient binning for `" $t "` with arbitrary width"]
            pub type [<Binning $t:upper>] = BinningWithWidth<$t>;
        }
        
        impl paste!{[<Binning $t:upper>]}{
            /// # Create a new Binning
            /// * both borders are inclusive
            /// * each bin has width bin_width
            /// # Panics
            /// * if `start` is smaller than `end_inclusive`
            /// * if bin_width <= 0
            /// # Err
            /// Result is Err if start and end are mismatched with the bin_width, i.e., 
            /// it is impossible to create the binning due to the integer nature of our types
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

                // Check if the requested bin_width makes sense
                let u_width = bin_width as <$t as HasUnsignedVersion>::Unsigned;
                let left = to_u(start);
                let right = to_u(end_inclusive);
                let res = ((right - left) % u_width + 1) % u_width;
                // now res needs to be 0 for this to be a valid config
                if res != 0{
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
                #[doc = "# Iterator over all the bins\n\
                Example: \
                \n\
                ```
                use sampling::histogram::" [<Binning $t:upper>] ";\n\
                let binning = " [<Binning $t:upper>] "::new_inclusive(2,7,2).unwrap();\n\
                let vec: Vec<_> = binning.multi_valued_bin_iter().collect();\n\
                assert_eq!(&vec, &[(2, 3), (4, 5), (6, 7)]);\n\
                ```"]
                #[inline]
                pub fn multi_valued_bin_iter(&self) -> impl Iterator<Item=($t, $t)>
                {
                    let width = self.bin_width;
                    BinModIterHelper::new_unchecked(self.start, self.end_inclusive, width)
                }
            }

            /// # The amount of bins -1
            /// * minus 1 because if the bins are of width 1 and are going over the entire range of the type,
            ///     then we cannot represent the number of bins as this type
            /// 
            /// # Example
            /// If we look at an u8 and the range from 0 to 255, then this is 256 bins, which 
            /// cannot be represented as u8. To combat this, I return bins - 1.
            /// This works, because we always have at least 1 bin
            #[inline(always)]
            pub fn bins_m1(&self) -> <$t as HasUnsignedVersion>::Unsigned{
                let left = to_u(self.start);
                let right = to_u(self.end_inclusive);

                (right - left) / to_u(self.bin_width)
            }

            /// # Get the respective bin in native unsigned
            #[inline(always)]
            pub fn get_bin_index_native<V: Borrow<$t>>(&self, val: V) -> Option<<$t as HasUnsignedVersion>::Unsigned>{
                let val = *val.borrow();
                if self.is_inside(val)
                {
                    let bin_width = self.bin_width as <$t as HasUnsignedVersion>::Unsigned;
                    let index = (to_u(val) - to_u(self.start)) / bin_width;
                    Some(index)
                } else {
                    None
                }
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
            ///     `usize`
            #[inline(always)]
            fn get_bin_index<V: Borrow<$t>>(&self, val: V) -> Option<usize>{
                self.get_bin_index_native(val)
                    .map(|v| v as usize)
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
                        to_u(*val) - to_u(self.end_inclusive)
                    };
                    dist as f64
                }
            }

            /// # Iterates over all bins
            /// * Note: This implementation uses a more efficient representations of the bins underneath,
            ///     but is capable of returning the bins in this representation on request
            /// * Note also that this `Binning`  implements another method for the bin borders, i.e., `multi_valued_bin_iter`.
            ///     Consider using that instead, as it is more efficient
            fn bin_iter(&self) -> Box<dyn Iterator<Item=Bin<$t>>>{
                let iter = self
                    .multi_valued_bin_iter()
                    .map(
                        |(left, right)| Bin::InclusiveInclusive(left, right)
                    );
                Box::new(iter)
            }
        }

        impl HistogramCombine for GenericHist<paste!{[<Binning $t:upper>]}, $t>
        {
            fn align<S>(&self, right: S)-> Result<usize, super::HistErrors>
                where S: Borrow<Self> {
                let self_bins = self.binning();
                let right_bins = right.borrow().binning();
                
                if self_bins.bin_width != right_bins.bin_width{
                    return Err(HistErrors::ModuloError);
                }
            
                let right_first_border = right_bins.first_border();
            
                let idx = self_bins.get_bin_index(right_first_border)
                    .ok_or(HistErrors::OutsideHist)?;
            
                // now we have the index, but we need to check the alignment!
                let width_u = self_bins.bin_width as <$t as HasUnsignedVersion>::Unsigned;
                let distance = to_u(right_first_border) - to_u(self_bins.first_border());
                let modulo = distance % width_u;
                if modulo != 0 {
                    return Err(HistErrors::Alignment);
                }
                Ok(idx)
            }
        
            fn encapsulating_hist<S>(hists: &[S]) -> Result<Self, super::HistErrors>
                where S: Borrow<Self> {
                if hists.is_empty(){
                    return Err(HistErrors::EmptySlice);
                }
                let first = hists[0].borrow().binning();
                let width = first.bin_width;
                let mut left = first.first_border();
                let mut right = first.last_border();
                for other in hists[1..].iter(){
                    let binning = other.borrow().binning();
                    if width != binning.bin_width{
                        return Err(HistErrors::ModuloError);
                    }
                    left = left.min(binning.first_border());
                    right = right.max(binning.last_border());
                }
                // now I first create the binning, then I check if all the intervals aligned properly
                let new_binning = <paste!{[<Binning $t:upper>]}>::new_inclusive(
                    left, 
                    right, 
                    width
                ).map_err(|_| HistErrors::ModuloError)?;
            
                let new_first_border = to_u(new_binning.first_border());
                let bin_width_u = width as <$t as HasUnsignedVersion>::Unsigned;
            
                for hist in hists.iter(){
                    let binning = hist.borrow().binning();
                    let distance = to_u(binning.first_border()) - new_first_border;
                    let modulo = distance % bin_width_u;
                    if modulo != 0 {
                        return Err(HistErrors::Alignment);
                    }
                }
                Ok(GenericHist::new(new_binning))
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

#[cfg(test)]
mod tests{
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64;

    use crate::{GenericHist, Histogram, HistogramVal};

    use super::*;

    #[test]
    fn extreme_vals()
    { 
        let binning = BinningU8::new_inclusive(250,255,2).unwrap();
        let vec: Vec<_> = binning.multi_valued_bin_iter().collect();
        assert_eq!(&vec, &[(250, 251), (252, 253), (254, 255)]);
        let _binning = BinningU8::new_inclusive(0,255,1).unwrap();
        let _binning = BinningU8::new_inclusive(0,255,2).unwrap();
    }

    #[test]
    fn other_binning_hist_test()
    {
        use crate::HistogramInt;

        fn check(left: u8, right: u8, bin_width: u8)
        {
            let mut rng = Pcg64::seed_from_u64(23984);

            let binning = BinningU8::new_inclusive(left, right, bin_width).unwrap();
            let mut inefficient_hist = HistogramInt::new_inclusive(
                binning.left(), 
                binning.right(), 
                binning.bins_m1() as usize + 1
            ).unwrap();
            let mut this_hist = GenericHist::new(binning);
    
            for _ in 0..1000{
                let num = rng.gen_range(0..=9);
                this_hist.count_val(num).unwrap();
                inefficient_hist.increment_quiet(num);
            }
    
            let hist = this_hist.hist();
            let other_hist = inefficient_hist.hist();
            assert_eq!(hist, other_hist);
        }
        check(0, 9, 2);
        check(0, 254, 1);
        check(0, 253, 2);

    }

    #[test]
    fn unit_test_distance()
    {
        // # bin width 1
        let binning = BinningI8::new_inclusive(-50, 50, 1)
            .unwrap();

        let mut dist = binning.distance(i8::MIN);
        for i in i8::MIN+1..-50{
            let new_dist = binning.distance(i);
            assert!(dist > new_dist);
            dist = new_dist;
        }
        for i in -50..=50{
            assert_eq!(binning.distance(i), 0.0);
        }
        dist = 0.0;
        for i in 51..=i8::MAX{
            let new_dist = binning.distance(i);
            assert!(dist < new_dist);
            dist = new_dist;
        }

        // # bin width 2
        let binning = BinningI8::new_inclusive(-50, 49, 2)
            .unwrap();
        let mut dist = binning.distance(i8::MIN);
        for i in i8::MIN+1..-50{
            let new_dist = binning.distance(i);
            assert!(dist > new_dist);
            dist = new_dist;
        }
        for i in -50..=49{
            assert_eq!(binning.distance(i), 0.0);
        }
        dist = 0.0;
        for i in 50..=i8::MAX{
            let new_dist = binning.distance(i);
            assert!(dist < new_dist);
            dist = new_dist;
        }
    }

    #[test]
    fn test_combining()
    {
        let binning_1 = BinningI16::new_inclusive(
            -10, 
            9, 
            2
        ).unwrap();
        let hist_1 = GenericHist::new(binning_1);

        let binning_2 = BinningI16::new_inclusive(
            10, 
            11, 
            2
        ).unwrap();
        let hist_2 = GenericHist::new(binning_2);

        let encapsulating = GenericHist::encapsulating_hist(
            &[
                &hist_1,
                &hist_2
            ]
        ).unwrap();

        assert_eq!(
            encapsulating.binning().first_border(),
            -10
        );
        assert_eq!(
            encapsulating.binning().last_border(),
            11
        );

        let binning_3 = BinningI16::new_inclusive(
            12, 
            15, 
            2
        ).unwrap();
        let hist_3 = GenericHist::new(binning_3);


        let encapsulating = GenericHist::encapsulating_hist(
            &[
                &hist_3,
                &hist_1
            ]
        ).unwrap();

        assert_eq!(
            encapsulating.binning().first_border(),
            -10
        );
        assert_eq!(
            encapsulating.binning().last_border(),
            15
        );

        let misaligned_binning = BinningI16::new_inclusive(
            -11, 
            -10, 
            2
        ).unwrap();
        let misaligned_hist = GenericHist::new(misaligned_binning);

        match GenericHist::encapsulating_hist(
            &[
                &hist_3,
                &hist_1,
                &hist_2,
                &misaligned_hist
            ]
            ){
            Ok(_) => panic!("Bug in code! This has to give the error variant!"),
            Err(err) => {
                assert_eq!(
                    HistErrors::ModuloError,
                    err
                );
            }
        };

        let binning_4 = BinningI16::new_inclusive(
            -12, 
            -11, 
            2
        ).unwrap();
        let hist_4 = GenericHist::new(binning_4);

        match GenericHist::encapsulating_hist(
            &[
                &hist_3,
                &hist_1,
                &hist_2,
                &misaligned_hist,
                &hist_4
            ]
            ){
            Ok(_) => panic!("Bug in code! This has to give the error variant!"),
            Err(err) => {
                assert_eq!(
                    HistErrors::Alignment,
                    err
                );
            }
        };
    }
}