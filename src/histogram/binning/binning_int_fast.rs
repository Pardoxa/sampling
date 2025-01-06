use std::{
    ops::{
        RangeInclusive, 
        //Shl
    },
    borrow::Borrow,
    num::NonZeroUsize
};
use paste::paste;
use crate::HistogramVal;

use super::{
    Binning,
    HasUnsignedVersion,
    to_u,
    from_u,
    Bin,
    HistogramPartition,
    HistErrors,
    HistogramCombine,
    GenericHist,
    Histogram
};
use num_bigint::BigUint;

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

/// Generic binning meant for any integer type
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct FastSingleIntBinning<T>{
    /// left bin border, inclusive
    start: T,
    /// right bin border, inclusive
    end_inclusive: T
}


macro_rules! impl_binning {
    (
        $t:ty
    ) => {
        
        
        paste::item! {
            #[doc = "# Checked multiply divide\n\
            The operation is: a * b / denominator.\n\n \
            However this function guards against an overflow of a * b. \n\n As long as the mathematical result of a * b / denominator \
            is representable as unsigned version of `<" $t " as HasUnsignedVersion>::Unsigned` then the mathematical answer is returned. Otherwise, None is returned\n\n ## Note: \n\n `denominator` is not allowed to be 0"]

            pub fn [< checked_mul_div_ $t >] (
                a: <$t as HasUnsignedVersion>::Unsigned, 
                b: <$t as HasUnsignedVersion>::Unsigned,
                denominator: <$t as HasUnsignedVersion>::Unsigned
            ) -> Option<<$t as HasUnsignedVersion>::Unsigned>
            {

                if let Some(val) = a.checked_mul(b){
                    return Some(val / denominator);
                }

                enum Answer{
                    Known(Option<<$t as HasUnsignedVersion>::Unsigned>),
                    Unknown
                }
                
                #[inline(always)]
                fn mul_div(
                    mut a: <$t as HasUnsignedVersion>::Unsigned, 
                    mut b: <$t as HasUnsignedVersion>::Unsigned, 
                    denominator: <$t as HasUnsignedVersion>::Unsigned
                ) -> Answer
                {
                    if a < b {
                        std::mem::swap(&mut a, &mut b);
                    }
                    // idea here: a / denominator * b + (a % denominator) * b / denominator
                    // if it works, this should be faster than the alternative.
                    // 
                    // If (a/denominator) * b overflows we know that the result cannot be represented by the type we want.
                    // If it does not overflow, this method works only if (a%denominator)*b does not overflow.
                    // Thus we check that first.
                    let left = match (a / denominator).checked_mul(b){
                        None => return Answer::Known(None),
                        Some(val) => val
                    };
                    let right_mul = match (a % denominator)
                        .checked_mul(b){
                            None => return Answer::Unknown,
                            Some(v) => v
                        };
                    
                    
                    let result = left.checked_add(right_mul / denominator);
                    Answer::Known(result)
                }

                match mul_div(a, b, denominator){
                    Answer::Known(res) => return res,
                    Answer::Unknown => {
                        let a: BigUint = a.into();
                        let b: BigUint = b.into();
                        let denominator: BigUint = denominator.into();
                        let res = a * b / denominator;
                        res.try_into().ok()
                    } 
                }

            }
        }
        
        paste!{
            #[doc = "Efficient binning for `" $t "` with bins of width 1"]
            pub type [<FastBinning $t:upper>] = FastSingleIntBinning<$t>;
        }

        impl paste!{[<FastBinning $t:upper>]}{
            /// # Create a new Binning
            /// * both borders are inclusive
            /// * each bin has width 1
            /// # Panics
            /// * if `start` is smaller than `end_inclusive`
            #[inline(always)]
            pub const fn new_inclusive(start: $t, end_inclusive: $t) -> Self{
                assert!(start <= end_inclusive, "Start needs to be <= end_inclusive!");
                Self {start, end_inclusive }
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
                use sampling::histogram::" [<FastBinning $t:upper>] ";\n\
                let binning = " [<FastBinning $t:upper>] "::new_inclusive(2,5);\n\
                let vec: Vec<_> = binning.single_valued_bin_iter().collect();\n\
                assert_eq!(&vec, &[2, 3, 4, 5]);\n\
                ```"]
                pub fn single_valued_bin_iter(&self) -> impl Iterator<Item=$t>
                {
                    self.range_inclusive()
                }
            }

            /// # The amount of bins -1
            /// * minus 1 because if the bins are going over the entire range of the type,
            ///     then I cannot represent the number of bins as this type
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

        impl paste!{[<FastBinning $t:upper>]}
        {
            /// # Get the respective bin index
            /// * Similar to get_bin_index, but without the cast to `usize`. This means that large types are not at a risk of overflow here
            #[inline(always)]
            pub fn get_bin_index_native<V: Borrow<$t>>(&self, val: V) -> Option<<$t as HasUnsignedVersion>::Unsigned>{
                let val = *val.borrow();
                if self.is_inside(val)
                {
                    Some(to_u(val) - to_u(self.start))
                } else{
                    None
                }
            }
        }

        impl GenericHist<paste!{[<FastBinning $t:upper>]}, $t>{
            /// # Iterate over bins and hits
            /// Returns an iterator, which gives yields (bin, hits), i.e.,
            /// a number that represents the bin (since the bin is of size 1)
            /// and the corresponding number of hits
            pub fn bin_hits_iter(&'_ self) -> impl Iterator<Item=($t, usize)> + '_
            {
                self.binning()
                    .single_valued_bin_iter()
                    .zip(self.hist().iter().copied())
            }
        }

 
        impl Binning<$t> for paste!{[<FastBinning $t:upper>]} {
            #[inline(always)]
            fn get_bin_len(&self) -> usize 
            {
                (self.bins_m1() as usize).saturating_add(1)
            }

            /// # Get the respective bin index
            /// * Note: Obviously this breaks when the bin index cannot be represented as `usize`, in that case Some(usize::MAX) will be returned
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
            #[inline(always)]
            fn first_border(&self) -> $t{
                self.start
            }

            #[inline(always)]
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
            #[inline(always)]
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
            /// * Note: This implementation use more efficient representations of the bins underneath,
            ///     but are capable of returning the bins in this representation on request
            /// * Note also that this `Binning`  implements another method for the bin borders, i.e., `single_valued_bin_iter`.
            ///     Consider using that instead, as it is more efficient
            fn bin_iter(&self) -> Box<dyn Iterator<Item=Bin<$t>>>{
                Box::new(
                    self.range_inclusive()
                        .map(|val| Bin::SingleValued(val))
                    )
            }
        }

        impl HistogramPartition for paste!{[<FastBinning $t:upper>]}
        {
            paste!{
                #[doc = "# partition the interval\
                \n* returns Vector of `n` Binnings. Though `n` will be limited by the max value that `" $t "` can hold.   \
                ## parameter \n\
                * `n` number of resulting intervals. \n\
                * `overlap` How much overlap should there be? \n\
                ## To understand overlap, we have to look at the formula for the i_th interval in the result vector: \n\
                let ``left`` be the left border of ``self`` and ``right`` be the right border of self \n\
                * left border of interval i = left + i * (right - left) / (n + overlap) \n\
                 * right border of interval i = left + (i + overlap) * (right - left) / (n + overlap) \n\
                ## What is it for? \
                \n * This is intended to create multiple overlapping intervals, e.g., for a Wang-Landau simulation\
                \n # Note\
                \n * Will fail if `overlap` + `n` are not representable as `" $t "`"]
                fn overlapping_partition(&self, n: NonZeroUsize, overlap: usize) -> Result<Vec<Self>, HistErrors>
                {
                    let mut result = Vec::with_capacity(n.get());
                    let right_minus_left = self.bins_m1();
                    let n_native = n.get() as <$t as HasUnsignedVersion>::Unsigned;
                    let overlap_native = overlap as <$t as HasUnsignedVersion>::Unsigned;
                    let denominator = n_native
                        .checked_add(overlap_native)
                        .ok_or(HistErrors::Overflow)?;
                    for c in 0..n_native {
                        let left_distance = paste::item! { [< checked_mul_div_ $t >] }(c, right_minus_left, denominator)
                            .ok_or(HistErrors::Overflow)?;
                        let left = to_u(self.start) + left_distance;

                        let right_sum = c.saturating_add(overlap_native)
                            .checked_add(1)
                            .ok_or(HistErrors::Overflow)?;

                        let right_distance = paste::item! { [< checked_mul_div_ $t >] }(right_sum, right_minus_left, denominator)
                            .ok_or(HistErrors::Overflow)?;
                        let right = to_u(self.start) + right_distance;

                        let left = from_u(left);
                        let right = from_u(right);
                    
                        result.push(Self::new_inclusive(left, right));
                    }
                    debug_assert_eq!(
                        self.start, 
                        result[0].start, 
                        "eq1"
                    );
                    debug_assert_eq!(
                        self.end_inclusive, 
                        result.last().unwrap().end_inclusive, 
                        "eq2"
                    );
                    Ok(result)
                }
            }
        }

        impl HistogramCombine for GenericHist<paste!{[<FastBinning $t:upper>]}, $t>
        {
            fn align<S>(&self, right: S)-> Result<usize, HistErrors>
                where S: Borrow<Self> {
                let right = right.borrow();
                
                self.get_bin_index(right.first_border())
            }
        
            fn encapsulating_hist<S>(hists: &[S]) -> Result<Self, HistErrors>
                where S: Borrow<Self> {
                if hists.is_empty(){
                    return Err(HistErrors::EmptySlice);
                }
                let first_binning = hists[0].borrow().binning();
                let mut left = first_binning.first_border();
                let mut right = first_binning.last_border();
                for other in hists[1..].iter()
                {
                    let binning = other.borrow().binning();
                    left = left.min(binning.first_border());
                    right = right.max(binning.last_border());
                
                }
                let outer_binning = <paste!{[<FastBinning $t:upper>]}>::new_inclusive(left, right);
                let hist = GenericHist::new(outer_binning);
                Ok(hist)
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
    i128,
    usize,
    isize
);

#[cfg(test)]
mod tests{
    use std::fmt::{Debug, Display};

    use crate::GenericHist;
    use super::*;
    use crate::histogram::*;
    use num_traits::{PrimInt, AsPrimitive};

    fn hist_test_generic_all_inside<T>(left: T, right: T)
    where FastSingleIntBinning::<T>: Binning::<T>,
        GenericHist::<FastSingleIntBinning::<T>, T>: Histogram,
        T: PrimInt,
        std::ops::RangeInclusive<T>: Iterator<Item=T>,
    {
        let binning = FastSingleIntBinning::<T>{start: left, end_inclusive: right};
        let mut hist = 
            GenericHist::<FastSingleIntBinning::<T>, T>::new(binning);
         
        for (id, i) in (left..=right).enumerate() {
            assert!(hist.is_inside(i));
            assert_eq!(hist.is_inside(i), !hist.not_inside(i));
            assert!(hist.get_bin_index(i).unwrap() == id);
            assert_eq!(hist.distance(i), 0.0);
            hist.count_val(i).unwrap();
        }
        assert_eq!(hist.bin_enum_iter().count(), hist.bin_count());   
    }

    #[test]
    fn hist_inside()
    {
        //hist_test_generic_all_inside(20usize, 31usize);
        //hist_test_generic_all_inside(-23isize, 31isize);
        hist_test_generic_all_inside(-23i16, 31);
        hist_test_generic_all_inside(1u8, 3u8);
        //hist_test_generic_all_inside(123u128, 300u128);
        //hist_test_generic_all_inside(-123i128, 300i128);
        hist_test_generic_all_inside(u8::MIN, u8::MAX);
        hist_test_generic_all_inside(i8::MIN, i8::MAX);
        hist_test_generic_all_inside(-100i8, 100i8);
    }

    fn hist_test_generic_all_outside_extensive<T>(left: T, right: T)
    where FastSingleIntBinning::<T>: Binning::<T>,
        GenericHist::<FastSingleIntBinning::<T>, T>: Histogram,
        T: PrimInt,
        std::ops::Range<T>: Iterator<Item=T>,
        std::ops::RangeInclusive<T>: Iterator<Item=T>,
    {
        let binning = FastSingleIntBinning::<T>{start: left, end_inclusive: right};
        let hist = 
            GenericHist::<FastSingleIntBinning::<T>, T>::new(binning);
         
        for i in T::min_value()..left {
            assert!(hist.not_inside(i));
            assert_eq!(hist.is_inside(i), !hist.not_inside(i));
            assert!(matches!(hist.get_bin_index(i), Err(HistErrors::OutsideHist)));
            assert!(hist.distance(i) > 0.0);
        }
        for i in right+T::one()..=T::max_value() {
            assert!(hist.not_inside(i));
            assert_eq!(hist.is_inside(i), !hist.not_inside(i));
            assert!(matches!(hist.get_bin_index(i), Err(HistErrors::OutsideHist)));
            assert!(hist.distance(i) > 0.0);
        }
        assert_eq!(hist.bin_enum_iter().count(), hist.bin_count()); 
    }

    fn binning_all_outside_extensive<T>(left: T, right: T)
    where FastSingleIntBinning::<T>: Binning::<T>,
        T: PrimInt + Display,
        std::ops::Range<T>: Iterator<Item=T>,
        std::ops::RangeInclusive<T>: Iterator<Item=T> + Debug,
        std::ops::RangeFrom<T>: Iterator<Item=T>,
    {
        let binning = FastSingleIntBinning::<T>{start: left, end_inclusive: right};
         
        let mut last_dist = None; 
        for i in T::min_value()..left {
            assert!(binning.not_inside(i));
            assert_eq!(binning.is_inside(i), !binning.not_inside(i));
            assert!(binning.get_bin_index(i).is_none());
            let dist = binning.distance(i);
            assert!(dist > 0.0);
            match last_dist{
                None => last_dist = Some(dist),
                Some(d) => {
                    assert!(d > dist);
                    assert_eq!(d - 1.0, dist);
                    last_dist = Some(dist);
                }
            }
        }
        if let Some(d) = last_dist
        {
            assert_eq!(d, 1.0);
        }
        
        last_dist = None;
        for (i, dist_counter) in (right+T::one()..=T::max_value()).zip(1_u64..) {
            assert!(binning.not_inside(i));
            assert_eq!(binning.is_inside(i), !binning.not_inside(i));
            assert!(binning.get_bin_index(i).is_none());
            let dist = binning.distance(i);
            assert!(dist > 0.0);
            println!("{i}, {:?}", right+T::one()..=T::max_value());
            assert_eq!(dist, dist_counter.as_());
            match last_dist{
                None => last_dist = Some(dist),
                Some(d) => {
                    assert!(d < dist);
                    last_dist = Some(dist);
                }
            }
        }

        let binning = FastSingleIntBinning::<T>{start: left, end_inclusive: left};
        assert_eq!(binning.get_bin_len(), 1);
        assert_eq!(binning.get_bin_index(left), Some(0));
    
    }

    #[test]
    fn hist_outside()
    {
        hist_test_generic_all_outside_extensive(10u8, 20_u8);
        hist_test_generic_all_outside_extensive(-100, 100_i8);
        hist_test_generic_all_outside_extensive(-100, 100_i16);
        hist_test_generic_all_outside_extensive(123, 299u16);
    }

    #[test]
    fn binning_outside()
    {
        println!("0");
        binning_all_outside_extensive(0u8, 0_u8);
        println!("2");
        binning_all_outside_extensive(10u8, 20_u8);
        binning_all_outside_extensive(-100, 100_i8);
        binning_all_outside_extensive(-100, 100_i16);
        binning_all_outside_extensive(123, 299u16);
        //binning_all_outside_extensive(1, usize::MAX -100);
    }

    #[test]
    fn check_mul_div()
    {
        fn check(a: u8, b: u8, denominator: u8) -> Option<u8>
        {
            (a as u128 * b as u128 / denominator as u128).try_into().ok()
        }

        for i in 0..255{
            for j in 0..255{
                for k in 1..255{
                    assert_eq!(
                        check(i,j,k),
                        checked_mul_div_u8(i,j,k),
                        "Error in {i} {j} {k}"
                    );
                }
            }
        }
    }

    #[test]
    fn mul_testing()
    {
        use rand_pcg::Pcg64Mcg;
        use rand::SeedableRng;
        use rand::distributions::Uniform;
        use rand::prelude::*;
        macro_rules! mul_t {
            (
                $t:ty, $o:ty
            ) => {
                
                paste::item!{ fn [< mul_tests_ $t >]()
                    {
                        let mut rng = Pcg64Mcg::seed_from_u64(314668);
                        let uni_one = Uniform::new_inclusive(1, $t::MAX);
                        let uni_all = Uniform::new_inclusive(0, $t::MAX);
                        let max = <$t as HasUnsignedVersion>::Unsigned::MAX.into();
                        for _ in 0..100 {
                            let a = uni_all.sample(&mut rng);
                            let b = uni_all.sample(&mut rng);
                            let c = uni_one.sample(&mut rng);
                            let result: $o = a as $o * b as $o / c as $o;
                            let mul = paste::item! { [< checked_mul_div_ $t >]}(
                                a as <$t as HasUnsignedVersion>::Unsigned,
                                b as <$t as HasUnsignedVersion>::Unsigned,
                                c as <$t as HasUnsignedVersion>::Unsigned
                            );
                            if result <= max {
                                assert_eq!(
                                    mul,
                                    Some(result as <$t as HasUnsignedVersion>::Unsigned)
                                )
                            } else {
                                assert!(mul.is_none());
                            }
                        }
                    }
                }
            }
        } 
        mul_t!(u8, u16);
        mul_tests_u8();
        mul_t!(u16, u64);
        mul_tests_u16();
        mul_t!(u32, u128);
        mul_tests_u32();
        mul_t!(i8, i16);
        mul_tests_i8();
        mul_t!(i32, i128);
        mul_tests_i32();
    }  
  
    
    
    #[test]
    fn partion_test()
    {
        let n = NonZeroUsize::new(2).unwrap();
        let h = FastBinningU8::new_inclusive(0, u8::MAX);
        for overlap in 0..10{
            let h_part = h.overlapping_partition(n, overlap).unwrap();
            assert_eq!(h.first_border(), h_part[0].first_border());
            assert_eq!(h.last_border(), h_part.last().unwrap().last_border());
        }



        let h = FastBinningI8::new_inclusive(i8::MIN, i8::MAX);
        let h_part = h.overlapping_partition(n, 0).unwrap();
        assert_eq!(h.first_border(), h_part[0].first_border());
        assert_eq!(h.last_border(), h_part.last().unwrap().last_border());

        let h = FastBinningI16::new_inclusive(i16::MIN, i16::MAX);
        let h_part = h.overlapping_partition(n, 2).unwrap();
        assert_eq!(h.first_border(), h_part[0].first_border());
        assert_eq!(h.last_border(), h_part.last().unwrap().last_border());


        let _ = h.overlapping_partition(NonZeroUsize::new(2000).unwrap(), 0).unwrap();
    }

    #[test]
    fn overlapping_partition_test2()
    {
        use rand_pcg::Pcg64Mcg;
        use rand::distributions::Uniform;
        use rand::prelude::*;
        let mut rng = Pcg64Mcg::seed_from_u64(2314668);
        let uni = Uniform::new_inclusive(-100, 100);
        for overlap in 0..=3 {
            for i in 0..100 {
                let (left, right) = loop {
                    let mut num_1 = uni.sample(&mut rng);
                    let mut num_2 = uni.sample(&mut rng);

                    if num_1 != num_2 {
                        if num_2 < num_1 {
                            std::mem::swap(&mut num_1, &mut num_2);
                        }
                        if (num_2 as isize - num_1 as isize) < (overlap as isize + 1) {
                            continue;
                        }
                        break (num_1, num_2)
                    }
                };
                println!("iteration {i}");
                let hist_fast = FastBinningI8::new_inclusive(left, right);
                let overlapping = hist_fast
                    .overlapping_partition(NonZeroUsize::new(3).unwrap(), overlap)
                    .unwrap();

                assert_eq!(
                    overlapping.last().unwrap().last_border(),
                    hist_fast.last_border(),
                    "overlapping_partition_test2 - last border check"
                );

                assert_eq!(
                    overlapping.first().unwrap().first_border(),
                    hist_fast.first_border(),
                    "overlapping_partition_test2 - first border check"
                );

                for slice in overlapping.windows(2){
                    assert!(
                        slice[0].first_border() <= slice[1].first_border()
                    );
                    assert!(
                        slice[0].last_border() <= slice[1].last_border()
                    );
                }
            }
        }
    }

    #[test]
    fn hist_combine()
    {
        let binning_left = FastBinningI8::new_inclusive(-5, 0);
        let binning_right = FastBinningI8::new_inclusive(-1, 2);
        let left = GenericHist::new(binning_left);
        let right = GenericHist::new(binning_right);

        let encapsulating = GenericHist::encapsulating_hist(&[&left, &right]).unwrap();
        let enc_binning = encapsulating.binning();
        assert_eq!(enc_binning.first_border(), binning_left.first_border());
        assert_eq!(enc_binning.last_border(), binning_right.last_border());
        assert_eq!(encapsulating.bin_count(), 8);

        let align = left.align(right).unwrap();

        assert_eq!(align, 4);

        let left = FastBinningI8::new_inclusive(i8::MIN, 0)
            .to_generic_hist();
        let right = FastBinningI8::new_inclusive(0, i8::MAX)
            .to_generic_hist();

        let en = GenericHist::encapsulating_hist(&[&left, &right]).unwrap();

        assert_eq!(en.bin_count(), 256);

        let align = left.align(right).unwrap();

        assert_eq!(128, align);

        let left = FastBinningI8::new_inclusive(i8::MIN, i8::MAX)
            .to_generic_hist();
        let small = FastBinningI8::new_inclusive(127, 127)
            .to_generic_hist();

        let align = left.align(&small).unwrap();

        assert_eq!(255, align);

        let en = GenericHist::encapsulating_hist(&[&left]).unwrap();
        assert_eq!(en.bin_count(), 256);
        let slice = [&left];
        let en = GenericHist::encapsulating_hist(&slice[1..]);
        assert_eq!(en.err(), Some(HistErrors::EmptySlice));
        let en = GenericHist::encapsulating_hist(&[small, left]).unwrap();

        assert_eq!(en.bin_count(), 256);
    }

    #[test]
    fn unit_test_distance()
    {
        // # bin width 1
        let binning = FastBinningI8::new_inclusive(-50, 50);

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
    }
}