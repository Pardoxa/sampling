use std::{
    ops::{
        RangeInclusive, 
        //Shl
    },
    borrow::Borrow
};
use paste::paste;
use super::{
    Binning,
    HasUnsignedVersion,
    to_u,
    //from_u,
    Bin,
    //HistogramPartition,
    //HistErrors
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


macro_rules! impl_binning {
    (
        $t:ty
    ) => {
        /* 
        paste::item! {
            fn [< checked_mul_div_ $t >] (
                mut a: <$t as HasUnsignedVersion>::Unsigned, 
                mut b: <$t as HasUnsignedVersion>::Unsigned,
                denominator: <$t as HasUnsignedVersion>::Unsigned
            ) -> Option<<$t as HasUnsignedVersion>::Unsigned>
            {
                if a == denominator{
                    return Some(b);
                } else if b == denominator {
                    return Some(a);
                }

                if let Some(val) = a.checked_mul(b){
                    return Some(val / denominator);
                }


                
                if a < b {
                    std::mem::swap(&mut a, &mut b);
                }
                // test, to change:

                return Some(
                    ((a as i128 * b as i128) / denominator as i128) as <$t as HasUnsignedVersion>::Unsigned
                );
                todo!("");
                // a >= b is now true
                let mut sum_rest = 0;
                if a >= denominator{
                    sum_rest += a / denominator;
                    a %= denominator;
                }
                println!("sum_rest {sum_rest} a: {a} b: {b} denominator {denominator}");
                let (overflows, rest) = paste::item! { [< overflow_counting_mul_ $t >] }(a, b);
                // (overflows * (num + 1) + rest) / denominator;
                // -> overflows * (num / denominator + 1/ denominator) + rest / denominator
                let mut num_den = <$t as HasUnsignedVersion>::Unsigned::MAX / denominator;
                let mut num_res = <$t as HasUnsignedVersion>::Unsigned::MAX % denominator;
                let num_res_p1 = num_res.checked_add(1)?;
                println!("here 1");
                num_den = num_den.checked_add(num_res_p1 / denominator)?;
                println!("here 2");
                num_res = num_res_p1 % denominator;
                println!("here 3");
                num_den = num_den.checked_mul(overflows)?;
                println!("here 4 num_den {num_den} overflows {overflows} num_res");
                let (new_overflows, res) = paste::item! { [< overflow_counting_mul_ $t >] }(num_res, overflows);
                num_res = res;
                // ich muss den overflow noch irgendwie einbeziehen -> new_overflow / denominator…
                // in der zeile hier drunter ist wahrscheinlich noch der wurm drin
                let overflow_rest = ((<$t as HasUnsignedVersion>::Unsigned::MAX % denominator) + 1) / denominator;
                num_den += new_overflows.checked_mul((<$t as HasUnsignedVersion>::Unsigned::MAX / denominator))?;
                num_den += overflow_rest;
                println!("here 5");
                num_den = num_den.checked_add(num_res / denominator)?;
                println!("here 6");
                num_res %= denominator;
                println!("here 7");
                num_res = num_res.checked_add(rest)?;
                println!("here 8");
                let result = num_den.checked_add(num_res / denominator)?;
                println!("result: {result}");
                result.checked_add(sum_rest.checked_mul(b)?)

            }
        }

        paste::item! {
            fn [< overflow_counting_mul_ $t >] (
                a: <$t as HasUnsignedVersion>::Unsigned, 
                b: <$t as HasUnsignedVersion>::Unsigned
            )  
                -> (<$t as HasUnsignedVersion>::Unsigned, <$t as HasUnsignedVersion>::Unsigned)
            {
                if let Some(res) = a.checked_mul(b)
                {
                    (0, res)
                } else{
                    #[inline]
                    fn check_bit_at(input: <$t as HasUnsignedVersion>::Unsigned, bit: <$t as HasUnsignedVersion>::Unsigned) -> bool {
                    
                        input & bit != 0

                    }
                    let mut bit = 1;
                    let mut sum: <$t as HasUnsignedVersion>::Unsigned = 0;
                    let mut overflow_counter = 0;
                    for i in 0..$t::BITS {
                        let mut shifted_num = b;
                        if check_bit_at(a, bit){
                            let mut to_shift = i;
                            let mut current_overflow_counter = 0;
                            loop {
                                let overflow;
                                if to_shift <= shifted_num.leading_zeros(){
                                    (sum, overflow) = sum.overflowing_add(shifted_num.shl(to_shift));
                                    overflow_counter += current_overflow_counter;
                                    if overflow {
                                        overflow_counter += 1;
                                    }
                                    break;
                                } else if shifted_num.leading_zeros() > 0{
                                    to_shift -= shifted_num.leading_zeros();
                                    shifted_num = shifted_num.shl(shifted_num.leading_zeros());
                                } else {
                                    shifted_num = shifted_num.shl(1);
                                    to_shift -= 1;
                                    current_overflow_counter = current_overflow_counter.shl(1);
                                    current_overflow_counter += 1;
                                }
                            }
                        }
                        bit = bit.shl(1);
                    }
                    (overflow_counter, sum)
    
                }
            }
        }*/
        
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

 
        impl Binning<$t> for paste!{[<FastBinning $t:upper>]} {
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

        /*impl HistogramPartition for paste!{[<FastBinning $t:upper>]}
        {
           
            /// # partition the interval
            /// * returns Vector of `n` Binnings that  
            /// ## parameter
            /// * `n` number of resulting intervals
            /// * `overlap` How much overlap should there be?
            /// ## To understand overlap, we have to look at the formula for the i_th interval in the result vector:
            ///
            ///    let ``left`` be the left border of ``self`` and ``right`` be the right border of self
            ///
            ///    * left border of interval i = left + i * (right - left) / (n + overlap)
            ///    * right border of interval i = left + (i + overlap) * (right - left) / (n + overlap)
            ///
            /// ## What is it for?
            /// * This is intended to create multiple overlapping intervals, e.g.,
            /// for a Wang-Landau simulation
            fn overlapping_partition(&self, n: usize, overlap: usize) -> Result<Vec<Self>, HistErrors>
            {
                dbg!(self, n, overlap);
                let mut result = Vec::with_capacity(n);
                let right_minus_left = self.bins_m1();
                let n_native = n as <$t as HasUnsignedVersion>::Unsigned;
                let denominator = (n + overlap) as <$t as HasUnsignedVersion>::Unsigned;
                let overlap_native = overlap as <$t as HasUnsignedVersion>::Unsigned;
                for c in 0..n_native {
                    println!("1");
                    let left_distance = match paste::item! { [< checked_mul_div_ $t >] }(c, right_minus_left, denominator)
                    {
                        Some(mul_res) => mul_res ,
                        None => { return Err(HistErrors::Overflow)}
                    };
                        println!("2");
                    let left = to_u(self.start) + left_distance;
                    println!("3");
                    
                    let right_sum = c.saturating_add(overlap_native)
                        .checked_add(1)
                        .ok_or(HistErrors::Overflow)?;

                    let right_distance = match  paste::item! { [< checked_mul_div_ $t >] }(right_sum, right_minus_left, denominator)
                    {
                        Some(mul_res) => mul_res ,
                        None => { return Err(HistErrors::Overflow)}
                    };
                    let right = to_u(self.start) + right_distance;

                    let left = from_u(left);
                    let right = from_u(right);
                    println!("left {left} right {right}");
                    println!("goal: {} {}", self.start, self.end_inclusive);
                
                    result.push(Self::new_inclusive(left, right));
                }
                dbg!(&result);
                assert_eq!(
                    self.start, 
                    result[0].start, 
                    "eq1"
                );
                assert_eq!(
                    self.end_inclusive, 
                    result.last().unwrap().end_inclusive, 
                    "eq2"
                );
                for (entry_old, entry_new) in result.iter().zip(result.iter().skip(1))
                {
                    assert!(
                        entry_old.start < entry_old.end_inclusive,
                        "Start needs to be smaller than end"
                    );
                    println!("entry_old.start {} <= {} entry_new.start", entry_old.end_inclusive, entry_new.start);
                    assert!(entry_old.start < entry_new.start);
                    println!("entry_old.end_inclusive {} <= {} entry_new.end_inclusive", entry_old.end_inclusive, entry_new.end_inclusive);
                    assert!(entry_old.end_inclusive < entry_new.end_inclusive);
                    assert!(entry_new.start < entry_new.end_inclusive);
                }
                Ok(result)
            }
        }*/
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
    // use rand_pcg::Pcg64Mcg;
    // use rand::SeedableRng;
    // use rand::distributions::Uniform;
    // use rand::prelude::*;
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

    /* 
    fn check_widening(a: u8, b: u8) 
    {
        let actual = (a as usize * b as usize) / 256;
        let rest = (a as usize * b as usize) % 256;
        let (m_a, m_r) = overflow_counting_mul_u8(a, b);
        println!("actual overflow {actual} my {m_a}");
        println!("actual rest {rest} mine {m_r}");
        assert_eq!(actual as u8, m_a);
        assert_eq!(rest as u8, m_r);
    }

    fn check_widening_u32(a: u32, b: u32) 
    {
        let actual = (a as u128 * b as u128) / (u32::MAX as u128 + 1);
        let rest = (a as u128 * b as u128) % (u32::MAX as u128 + 1);
        let (m_a, m_r) = overflow_counting_mul_u32(a, b);
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
        check_widening(128, 3);
        check_widening_u32(u32::MAX/2+1, 3);
    }*/

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

    /* 
    macro_rules! mul_t {
        (
            $t:ty, $o:ty
        ) => {
            
            paste::item!{ fn [< mul_tests_ $t >]()
                {
                    let mut rng = Pcg64Mcg::seed_from_u64(314668);
                    let uni_one = Uniform::new_inclusive(1, $t::max_value());
                    for _ in 0..100 {
                        let a = uni_one.sample(&mut rng);
                        let b = uni_one.sample(&mut rng);
                        let c = uni_one.sample(&mut rng);
                        let result: $o = a as $o * b as $o / c as $o;
                        let max = $t::max_value().into();
                        let mul = paste::item! { [< checked_mul_div_ $t >]}(a,b,c);
                        if result <= max {
                            println!("{a} {b} {c}");
                            println!("{mul:?} {result}");
                            assert!(mul.is_some());
                        } else {
                            assert!(mul.is_none());
                        }
                    }
                }
            }
        }
    } 

    #[test]
    fn mul_testing()
    {
        mul_t!(u8, u16);
        mul_tests_u8();
        
    }  
    */
    
    /* 
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
                let overlapping = hist_fast.overlapping_partition(3, overlap).unwrap();

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
            }
        }
    }*/
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