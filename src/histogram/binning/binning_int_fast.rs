use std::{
    ops::{RangeInclusive, Shl},
    borrow::Borrow
};
use paste::paste;
use super::{
    Binning,
    HasUnsignedVersion,
    to_u,
    from_u,
    Bin,
    HistogramPartition,
    HistErrors
};

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

pub fn widening_test(a: u8, b: u8) -> (u8, u8)
{
    if let Some(res) = a.checked_mul(b)
    {
        (0, res)
    } else{
        fn check_bit_at(input: u8, bit: u8) -> bool {
    
            input & bit != 0
            
        }
        let mut bit = 1;
        let mut sum = 0_u8;
        let mut overflow_counter = 0_u8;
        for i in 0..8 {
            let mut shifted_num = b;
            if check_bit_at(a, bit){
                println!("bit 1");
                let mut to_shift = i;
                let mut current_overflow_counter = 0_u8;
                loop {
                    let overflow;
                    println!("sum: {sum}");
                    println!("leading zeros {}", shifted_num.leading_zeros());
                    if to_shift <= shifted_num.leading_zeros(){
                        println!("to_shift <=  shifted_num.leading_zeros()");
                        println!("before shift {shifted_num}");
                        println!("shifted {} {to_shift}", shifted_num.shl(to_shift));
                        (sum, overflow) = sum.overflowing_add(shifted_num.shl(to_shift));
                        overflow_counter += current_overflow_counter;
                        if overflow {
                            overflow_counter += 1;
                        }
                        break;
                    } else if shifted_num.leading_zeros() > 0{
                        println!("shifted_num.leading_zeros() > 0");
                        shifted_num = shifted_num.shl(shifted_num.leading_zeros());
                        to_shift -= shifted_num.leading_zeros();
                    } else {
                        shifted_num = shifted_num.shl(1);
                        to_shift -= 1;
                        println!("Here");
                        current_overflow_counter = current_overflow_counter.shl(1);
                        current_overflow_counter += 1;
                    }
                }
            } else {
                println!("bit 0");
            }
            println!("sum_end: {sum}");
            bit = bit.shl(1);
        }
        (overflow_counter, sum)
    }
}

macro_rules! impl_binning {
    (
        $t:ty
    ) => {
        paste::item! {
            fn [< widening_mul_ $t:upper >] (mut a: <$t as HasUnsignedVersion>::Unsigned, mut b: <$t as HasUnsignedVersion>::Unsigned)  
                -> (<$t as HasUnsignedVersion>::Unsigned, <$t as HasUnsignedVersion>::Unsigned)
            {
                if let Some(res) = a.checked_mul(b)
                {
                    (0, res)
                } else{
                    fn check_bit_at(input: <$t as HasUnsignedVersion>::Unsigned, bit: <$t as HasUnsignedVersion>::Unsigned) -> bool {
                    
                        input & bit != 0

                    }
                    let mut bit = 1;
                    let mut sum: <$t as HasUnsignedVersion>::Unsigned = 0;
                    let mut overflow_counter = 0;
                    for i in 0..8 {
                        let mut shifted_num = b;
                        if check_bit_at(a, bit){
                            println!("bit 1");
                            let mut to_shift = i;
                            let mut current_overflow_counter = 0;
                            loop {
                                let overflow;
                                println!("sum: {sum}");
                                println!("leading zeros {}", shifted_num.leading_zeros());
                                if to_shift <= shifted_num.leading_zeros(){
                                    println!("to_shift <=  shifted_num.leading_zeros()");
                                    println!("before shift {shifted_num}");
                                    println!("shifted {} {to_shift}", shifted_num.shl(to_shift));
                                    (sum, overflow) = sum.overflowing_add(shifted_num.shl(to_shift));
                                    overflow_counter += current_overflow_counter;
                                    if overflow {
                                        overflow_counter += 1;
                                    }
                                    break;
                                } else if shifted_num.leading_zeros() > 0{
                                    println!("shifted_num.leading_zeros() > 0");
                                    shifted_num = shifted_num.shl(shifted_num.leading_zeros());
                                    to_shift -= shifted_num.leading_zeros();
                                } else {
                                    shifted_num = shifted_num.shl(1);
                                    to_shift -= 1;
                                    println!("Here");
                                    current_overflow_counter = current_overflow_counter.shl(1);
                                    current_overflow_counter += 1;
                                }
                                println!("overflow counter: {current_overflow_counter}");
                            }
                        } else {
                            println!("bit 0");
                        }
                        println!("sum_end: {sum}");
                        bit = bit.shl(1);
                    }
                    (overflow_counter, sum)
    
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
            /// TODO Think about if this should actually panic or return None
            #[inline(always)]
            pub const fn new_inclusive(start: $t, end_inclusive: $t) -> Self{
                assert!(start <= end_inclusive);
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
                Box::new(
                    self.range_inclusive()
                        .map(|val| Bin::SingleValued(val))
                    )
            }
        }

        impl HistogramPartition for paste!{[<FastBinning $t:upper>]}
        {
           
            /// # partition the interval
            /// * returns Vector of `n` Binnings that  
            /// ## parameter
            /// * `n` number of resulting intervals
            /// * `overlap` How much overlap should there be?
            /// ## What is it for?
            /// * This is intended to create multiple overlapping intervals, e.g.,
            /// for a Wang-Landau simulation
            fn overlapping_partition(&self, n: usize, overlap: usize) -> Result<Vec<Self>, HistErrors>
            {
                let mut result = Vec::with_capacity(n);
                // TODO maybe change n: usize to n: Unsigned?
                let size = self.bins_m1();
                let n_native = n as <$t as HasUnsignedVersion>::Unsigned;
                let denominator = (n + overlap) as <$t as HasUnsignedVersion>::Unsigned;
                let overlap_native = overlap as <$t as HasUnsignedVersion>::Unsigned;
                for c in 0..n_native {
                    println!("1");
                    let left_distance = c.checked_mul(size)
                        .ok_or(HistErrors::Overflow)?
                        / denominator;
                        println!("2");
                    let left = to_u(self.start) + left_distance;
                    println!("3");
                    /*
                        Idea: I can calculate the number of overflows beforehand?
                        Maybe.
                        But how?
                        I do not want to use something other than u8- otherwise it would be simple.
                        a*b/256…~ (128*2)*(128*2) --- need to know power of two…
                        bits: 2^0+2^1 usw.
                        (2^0+2^3)(2^0+2^3)-> (2^0+2^3+2^3+2^(3+3))/(2^7)
                        alles kleiner 2^7 ist also erstmal egal… aber das wird viel zu lang…


                        Neuer ansatz
                        (a*b)/256 -> log(a*b/256) ) log(a)+log(b)-log(256)
                     */
                    let right_distance = (c + overlap_native + 1)
                        .checked_mul(size)
                        .ok_or(HistErrors::Overflow)?
                        / denominator;
                        println!("4");
                    let right = to_u(self.start) + right_distance;

                    let left = from_u(left);
                    let right = from_u(right);
                
                    result.push(Self::new_inclusive(left, right));
                }
                Ok(result)
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
    use std::fmt::{Display, Debug};

    use crate::GenericHist;
    use rand_pcg::Pcg64Mcg;
    use rand::SeedableRng;
    use rand::distributions::Uniform;
    use rand::prelude::*;
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

    fn check_widening(a: u8, b: u8) 
    {
        let actual = (a as usize * b as usize) / 256;
        let rest = (a as usize * b as usize) % 256;
        let (m_a, m_r) = widening_mul_U8(a, b);
        println!("actual overflow {actual} my {m_a}");
        println!("actual rest {rest} mine {m_r}");
        assert_eq!(actual as u8, m_a);
        assert_eq!(rest as u8, m_r);
    }

    fn check_widening_u32(a: u32, b: u32) 
    {
        let actual = (a as u128 * b as u128) / (u32::MAX as u128 + 1);
        let rest = (a as u128 * b as u128) % (u32::MAX as u128 + 1);
        let (m_a, m_r) = widening_mul_U32(a, b);
        println!("actual overflow {actual} my {m_a}");
        println!("actual rest {rest} mine {m_r}");
        assert_eq!(actual as u32, m_a);
        assert_eq!(rest as u32, m_r);
    }

    #[test]
    fn widening_testing()
    {
        check_widening(1, 255);
        check_widening(3, 128);
        check_widening(2, 255);
        check_widening(255, 255);
        check_widening_u32(3, u32::MAX);
        check_widening(255, 3);
        check_widening_u32(u32::MAX, 3);
    }

    #[test]
    fn hist_inside()
    {
        hist_test_generic_all_inside(20usize, 31usize);
        hist_test_generic_all_inside(-23isize, 31isize);
        hist_test_generic_all_inside(-23i16, 31);
        hist_test_generic_all_inside(1u8, 3u8);
        hist_test_generic_all_inside(123u128, 300u128);
        hist_test_generic_all_inside(-123i128, 300i128);
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
            assert!(matches!(binning.get_bin_index(i), None));
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
            assert!(matches!(binning.get_bin_index(i), None));
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
        binning_all_outside_extensive(1, usize::MAX -100);
    }

    
      
    #[test]
    fn partion_test()
    {
        let h = FastBinningU8::new_inclusive(0, u8::MAX);
        let h_part = h.overlapping_partition(2, 0).unwrap();
        assert_eq!(h.first_border(), h_part[0].first_border());
        assert_eq!(h.last_border(), h_part.last().unwrap().last_border());


        let h = FastBinningI8::new_inclusive(i8::MIN, i8::MAX);
        let h_part = h.overlapping_partition(2, 0).unwrap();
        assert_eq!(h.first_border(), h_part[0].first_border());
        assert_eq!(h.last_border(), h_part.last().unwrap().last_border());

        let h = FastBinningI16::new_inclusive(i16::MIN, i16::MAX);
        let h_part = h.overlapping_partition(2, 2).unwrap();
        assert_eq!(h.first_border(), h_part[0].first_border());
        assert_eq!(h.last_border(), h_part.last().unwrap().last_border());


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
                let hist_fast = FastBinningI8::new_inclusive(left, right);
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
/*Below tests test a functionality that is not yet implemented
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
*/
}