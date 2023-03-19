
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
            #[inline(always)]
            pub fn new(start: $t, end_inclusive: $t) -> Self{
                assert!(start <= end_inclusive);
                Self {start, end_inclusive }
            }

            /// Get left border, inclusive
            #[inline(always)]
            pub fn left(&self) -> $t {
                self.start
            }

            /// Get right border, inclusive
            #[inline(always)]
            pub fn right(&self) -> $t
            {
                self.end_inclusive
            }

            /// # Returns the range covered by the bins as a `RangeInclusive<T>`
            #[inline(always)]
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
                self.bins_m1() as usize + 1
            }

            fn get_bin_index<V: Borrow<$t>>(&self, val: V) -> Option<usize>{
                let val = *val.borrow();
                if self.is_inside(val)
                {
                    Some((to_u(val) - to_u(self.start)) as usize)
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
            /// * any invalid numbers (like NAN or INFINITY) should have the highest distance possible
            /// * if a value corresponds to a valid bin, the distance should be zero
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

            fn borders_clone(&self) -> Result<Vec<$t>, HistErrors>{
                Ok(
                    self.range_inclusive()
                        .collect()
                )
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



#[cfg(test)]
mod tests{
    use crate::GenericHist;

    use super::*;
    use num_traits::PrimInt;
    use rand::{distributions::*, SeedableRng};
    use rand_pcg::Pcg64Mcg;

    fn hist_test_fast<T, B>(left: T, right: T)
    where GenericHist<B, T>: 
    {
        let mut hist = GenericHist::<B, T>::new_inclusive(left, right).unwrap();
        assert!(hist.not_inside(T::max_value()));
        assert!(hist.not_inside(T::min_value()));
        let two = unsafe{NonZeroUsize::new_unchecked(2)};
        for (id, i) in (left..=right).enumerate() {
            assert!(hist.is_inside(i));
            assert_eq!(hist.is_inside(i), !hist.not_inside(i));
            assert!(hist.get_bin_index(i).unwrap() == id);
            assert_eq!(hist.distance(i), 0.0);
            assert_eq!(hist.interval_distance_overlap(i, two), 0);
            hist.count_val(i).unwrap();
        }
        let lm1 = left - T::one();
        let rp1 = right + T::one();
        assert!(hist.not_inside(lm1));
        assert!(hist.not_inside(rp1));
        assert_eq!(hist.is_inside(lm1), !hist.not_inside(lm1));
        assert_eq!(hist.is_inside(rp1), !hist.not_inside(rp1));
        assert_eq!(hist.distance(lm1), 1.0);
        assert_eq!(hist.distance(rp1), 1.0);
        let one = unsafe{NonZeroUsize::new_unchecked(1)};
        assert_eq!(hist.interval_distance_overlap(rp1, one), 1);
        assert_eq!(hist.interval_distance_overlap(lm1, one), 1);
        assert_eq!(hist.borders_clone().unwrap().len() - 1, hist.bin_count());
    }

    #[test]
    fn hist_fast()
    {
        hist_test_fast(20usize, 31usize);
        hist_test_fast(-23isize, 31isize);
        hist_test_fast(-23i16, 31);
        hist_test_fast(1u8, 3u8);
        hist_test_fast(123u128, 300u128);
        hist_test_fast(-123i128, 300i128);

        hist_test_fast(-100i8, 100i8);
    }

    

    #[test]
    fn hist_creation()
    {
        let _ = HistU8Fast::new_inclusive(0, u8::MAX).unwrap();
        let _ = HistI8Fast::new_inclusive(i8::MIN, i8::MAX).unwrap();
    }

    #[test]
    fn partion_test()
    {
        let h = HistU8Fast::new_inclusive(0, u8::MAX).unwrap();
        let h_part = h.overlapping_partition(2, 0).unwrap();
        assert_eq!(h.left, h_part[0].left);
        assert_eq!(h.right, h_part.last().unwrap().right);


        let h = HistI8Fast::new_inclusive(i8::MIN, i8::MAX).unwrap();
        let h_part = h.overlapping_partition(2, 0).unwrap();
        assert_eq!(h.left, h_part[0].left);
        assert_eq!(h.right, h_part.last().unwrap().right);

        let h = HistI16Fast::new_inclusive(i16::MIN, i16::MAX).unwrap();
        let h_part = h.overlapping_partition(2, 2).unwrap();
        assert_eq!(h.left, h_part[0].left);
        assert_eq!(h.right, h_part.last().unwrap().right);


        let _ = h.overlapping_partition(2000, 0).unwrap();
    }

    #[test]
    fn overlapping_partition_test2()
    {
        let mut rng = Pcg64Mcg::seed_from_u64(2314668);
        let uni = Uniform::new_inclusive(-100, 100);
        for overlap in 0..=3 {
            for _ in 0..100 {
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
                let hist_fast = HistI8Fast::new_inclusive(left, right).unwrap();
                let overlapping = hist_fast.overlapping_partition(3, overlap).unwrap();

                assert_eq!(
                    overlapping.last().unwrap().last_border(),
                    hist_fast.last_border()
                );

                assert_eq!(
                    overlapping.first().unwrap().first_border(),
                    hist_fast.first_border()
                );
            }
        }
    }

    #[test]
    fn hist_combine()
    {
        let left = HistI8Fast::new_inclusive(-5,0).unwrap();
        let right = HistI8Fast::new_inclusive(-1, 2).unwrap();

        let en = HistI8Fast::encapsulating_hist(&[&left, &right]).unwrap();

        assert_eq!(en.left, left.left);
        assert_eq!(en.right, right.right);
        assert_eq!(en.bin_count(), 8);

        let align = left.align(right).unwrap();

        assert_eq!(align, 4);

        let left = HistI8Fast::new_inclusive(i8::MIN, 0).unwrap();
        let right = HistI8Fast::new_inclusive(0, i8::MAX).unwrap();

        let en = HistI8Fast::encapsulating_hist(&[&left, &right]).unwrap();

        assert_eq!(en.bin_count(), 256);

        let align = left.align(right).unwrap();

        assert_eq!(128, align);

        let left = HistI8Fast::new_inclusive(i8::MIN, i8::MAX).unwrap();
        let small = HistI8Fast::new_inclusive(127, 127).unwrap();

        let align = left.align(&small).unwrap();

        assert_eq!(255, align);

        let en = HistI8Fast::encapsulating_hist(&[&left]).unwrap();
        assert_eq!(en.bin_count(), 256);
        let slice = [&left];
        let en = HistI8Fast::encapsulating_hist(&slice[1..]);
        assert_eq!(en.err(), Some(HistErrors::EmptySlice));
        let en = HistI8Fast::encapsulating_hist(&[small, left]).unwrap();

        assert_eq!(en.bin_count(), 256);
    }

    #[test]
    fn hist_try_add()
    {
        let mut first = HistU8Fast::new_inclusive(0, 23)
            .unwrap();
        let mut second = HistU8Fast::new_inclusive(0, 23)
            .unwrap();
        
        for i in 0..=23{
            first.increment(i)
                .unwrap();
        }
        for i in 0..=11{
            second.increment(i)
                .unwrap();
        }

        first.try_add(&second)
            .unwrap();

        let hist = first.hist();

        #[allow(clippy::needless_range_loop)]
        for i in 0..=11{
            assert_eq!(hist[i], 2);
        }
        #[allow(clippy::needless_range_loop)]
        for i in 12..=23{
            assert_eq!(hist[i], 1);
        }

        let third = HistU8Fast::new(0,23)
            .unwrap();
            
        first.try_add(&third)
            .expect_err("Needs to be Err because ranges do not match");
    }

}