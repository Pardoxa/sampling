use{
    crate::histogram::*,
    num_traits::{
        ops::{
            checked::*, 
            wrapping::*
        },
        cast::*,
        identities::*,
        Bounded
    },
    std::{
        borrow::*,
        ops::*,
        num::*,
        sync::atomic::*
    }
};

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};


/// # Generic Histogram for integer types
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct HistogramInt<T>
{
    pub(crate) bin_borders: Vec<T>,
    pub(crate) hist: Vec<usize>,
}

impl<T> From<AtomicHistogramInt<T>> for HistogramInt<T>
{
    fn from(other: AtomicHistogramInt<T>) -> Self
    {
        let hist = other.hist
            .into_iter()
            .map(AtomicUsize::into_inner)
            .collect();
        Self{
            hist, 
            bin_borders: other.bin_borders
        }
    }
}

impl<T> HistogramInt<T>{
    /// similar to `self.borders_clone` but does not allocate memory
    pub fn borders(&self) -> &Vec<T>
    {
        &self.bin_borders
    }

    /// # Iterator over all the bins
    /// In HistogramInt a bin is defined by two values: The left border (inclusive)
    /// and the right border (exclusive)
    /// 
    /// Here you get an iterator which iterates over said borders.
    /// The Iterator returns a borrowed Array of length two, where the first value is 
    /// the left (inclusive) border and the second value is the right (exclusive) border
    /// 
    /// ## Example
    /// ```
    /// use sampling::histogram::*;
    /// 
    /// let hist = HistI8::new(0, 8, 4).unwrap();
    /// let mut bin_iter = hist.bin_iter();
    ///
    /// assert_eq!(bin_iter.next(), Some(&[0_i8, 2]));
    /// assert_eq!(bin_iter.next(), Some(&[2, 4]));
    /// assert_eq!(bin_iter.next(), Some(&[4, 6]));
    /// assert_eq!(bin_iter.next(), Some(&[6, 8]));
    /// assert_eq!(bin_iter.next(), None);
    /// ```
    pub fn bin_iter(& self) -> impl Iterator<Item = &[T;2]>
    {
        BorderWindow::new(self.bin_borders.as_slice())
    }

    /// # Iterate over all bins
    /// In HistogramInt a bin is defined by two values: The left border (inclusive)
    /// and the right border (exclusive)
    /// 
    /// This iterates over these values as well as the corresponding hit count (i.e. how often 
    /// the corresponding bin was hit)
    /// ## Item of Iterator
    /// `(&[left_border, right_border], number_of_hits)`
    /// ## Example
    /// ``` 
    /// use sampling::histogram::*;
    /// 
    /// let mut hist = HistUsize::new(0, 6, 3).unwrap();
    /// 
    /// hist.increment(0).unwrap();
    /// hist.increment(5).unwrap();
    /// hist.increment(4).unwrap();
    /// 
    /// let mut iter = hist.bin_hits_iter();
    /// assert_eq!(
    ///     iter.next(),
    ///     Some(
    ///         (&[0, 2], 1)
    ///     )
    /// );
    /// assert_eq!(
    ///     iter.next(),
    ///     Some(
    ///         (&[2, 4], 0)
    ///     )
    /// );
    /// assert_eq!(
    ///     iter.next(),
    ///     Some(
    ///         (&[4, 6], 2)
    ///     )
    /// );
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn bin_hits_iter(&self) -> impl Iterator<Item = (&[T;2], usize)>
    {
        self.bin_iter()
            .zip(
                self.hist
                    .iter()
                    .copied()
            )
    }

}


impl<T> HistogramInt<T>
where T: Sub<T, Output=T> + Add<T, Output=T> + Ord + One + Copy + NumCast
{
    #[inline]
    /// # Increment hit count 
    /// If `val` is inside the histogram, the corresponding bin count will be increased
    /// by 1 and the index corresponding to the bin in returned: `Ok(index)`.
    /// Otherwise an Error is returned
    /// ## Note
    /// This is the same as [HistogramVal::count_val]
    pub fn increment<V: Borrow<T>>(&mut self, val: V) -> Result<usize, HistErrors> {
        self.count_val(val)
    }

    #[inline]
    /// # Increment hit count
    /// Increments the hit count of the bin corresponding to `val`.
    /// If no bin corresponding to `val` exists, nothing happens
    pub fn increment_quiet<V: Borrow<T>>(&mut self, val: V)
    {
        let _ = self.increment(val);
    }
}


impl<T> HistogramInt<T>
where T: Copy{
    fn get_right(&self) -> T
    {
        self.bin_borders[self.bin_borders.len() - 1]
    }
}


impl<T> HistogramInt<T> 
where T: PartialOrd + ToPrimitive + FromPrimitive + CheckedAdd + One + HasUnsignedVersion + Bounded
        + Sub<T, Output=T> + Mul<T, Output=T> + Zero + Copy,
    std::ops::RangeInclusive<T>: Iterator<Item=T>,
    T::Unsigned: Bounded + HasUnsignedVersion<LeBytes=T::LeBytes, Unsigned=T::Unsigned> 
        + WrappingAdd + ToPrimitive + Sub<Output=T::Unsigned>
        + std::ops::Rem<Output=T::Unsigned> + FromPrimitive + Zero
        + std::cmp::Eq + std::ops::Div<Output=T::Unsigned>
        + Ord + std::ops::Mul<Output=T::Unsigned> + WrappingSub + Copy,
    std::ops::RangeInclusive<T::Unsigned>: Iterator<Item=T::Unsigned>
{
    /// # Create a new histogram
    /// * `right`: exclusive border
    /// * `left`: inclusive border
    /// * `bins`: how many bins do you need?
    /// # Note
    /// * `(right - left) % bins == 0` has to be true, otherwise
    ///     the bins cannot all have the same length!
    pub fn new(left: T, right: T, bins: usize) -> Result<Self, HistErrors> {
        if left >= right {
            return Err(HistErrors::IntervalWidthZero);
        } else if bins == 0 {
            return Err(HistErrors::NoBins);
        }
        let left_u = to_u(left);
        let right_u = to_u(right);
        let border_difference = right_u - left_u;
        let b = match T::Unsigned::from_usize(bins)
        {
            Some(val) => val,
            None => return Err(HistErrors::IntervalWidthZero),
        };
        if border_difference % b != T::Unsigned::zero() {
            return Err(HistErrors::ModuloError);
        }

        let bin_size = border_difference / b;

        if bin_size <= T::Unsigned::zero() {
            return Err(HistErrors::IntervalWidthZero);
        }
        
        let hist = vec![0; bins];
        let bin_borders: Vec<_> = (T::Unsigned::zero()..=b)
            .map(|val| {
                from_u(
                    left_u + to_u(val) * bin_size
                )
            })
            .collect();
        Ok(
            Self{
                bin_borders,
                hist
            }
        )
    }
    /// # Create a new histogram
    /// * equivalent to [`Self::new(left, right + 1, bins)`](#method.new)
    ///     (except that this method checks for possible overflow)
    /// # Note:
    /// * Due to implementation details, `right` cannot be `T::MAX` - 
    ///     if you try, you will get `Err(HistErrors::Overflow)`
    pub fn new_inclusive(left: T, right: T, bins: usize) -> Result<Self, HistErrors>
    {
        let right = match right.checked_add(&T::one()){
            None => return Err(HistErrors::Overflow),
            Some(val) => val,
        };
        Self::new(left, right, bins)
    }
}

impl<T> Histogram for HistogramInt<T>
{
    #[inline]
    fn bin_count(&self) -> usize {
        self.hist.len()
    }

    #[inline]
    fn hist(&self) -> &Vec<usize> {
        &self.hist
    }

    #[inline]
    fn increment_index_by(&mut self, index: usize, count: usize) -> Result<(), HistErrors> {
        match self.hist.get_mut(index) {
            None => Err(HistErrors::OutsideHist),
            Some(val) => {
                *val += count;
                Ok(())
            },
        }
    }

    #[inline]
    fn reset(&mut self) {
        // compiles down to memset :)
        self.hist
            .iter_mut()
            .for_each(|h| *h = 0);
    }
}

impl<T> HistogramVal<T> for HistogramInt<T>
where T: Ord + Sub<T, Output=T> + Add<T, Output=T> + One + NumCast + Copy
{
    fn count_val<V: Borrow<T>>(&mut self, val: V) -> Result<usize, HistErrors>
    {
        let id = self.get_bin_index(val)?;
        self.increment_index(id)
            .map(|_| id)
    }

    fn distance<V: Borrow<T>>(&self, val: V) -> f64 {
        let val = val.borrow();
        if self.not_inside(val) {
            let dist = if *val < self.first_border() {
                self.first_border() - *val
            } else {
                *val - self.get_right() + T::one()
            };
            dist.to_f64().unwrap()
        } else {
            0.0
        }
    }

    #[inline]
    fn first_border(&self) -> T {
        self.bin_borders[0]
    }

    fn last_border(&self) -> T {
        self.bin_borders[self.bin_borders.len() - 1]
    }

    #[inline(always)]
    fn last_border_is_inclusive(&self) -> bool {
        false
    }

    #[inline]
    fn is_inside<V: Borrow<T>>(&self, val: V) -> bool {
        let val = *val.borrow();
        val >= self.first_border()
            && val < self.get_right()
    }

    #[inline]
    fn not_inside<V: Borrow<T>>(&self, val: V) -> bool {
        let val = *val.borrow();
        val < self.first_border()
            || val >= self.get_right()
    }

    /// None if not inside Hist covered zone
    fn get_bin_index<V: Borrow<T>>(&self, val: V) -> Result<usize, HistErrors>
    {
        let val = val.borrow();
        if self.not_inside(val)
        {
            return Err(HistErrors::OutsideHist);
        }

        self.bin_borders
            .binary_search(val.borrow())
            .or_else(|index_m1| Ok(index_m1 - 1))
    }

    /// # consider using `self.bin_iter()` instead
    /// * this will return an iterator over the bins for displaying purposes
    /// * all bins are defined via an inclusive and an exclusive border
    /// * It is more efficient to use `self.bin_iter()`instead
    fn bin_enum_iter(&'_ self) -> Box<dyn Iterator<Item=Bin<T>> + '_> {
        let iter = self.bin_iter()
            .map(|[left, right]| Bin::InclusiveExclusive(*left, *right));
        Box::new(iter)
    }
}

impl<T> HistogramIntervalDistance<T> for HistogramInt<T> 
where T: Ord + Sub<T, Output=T> + Add<T, Output=T> + One + NumCast + Copy
{
    fn interval_distance_overlap<V: Borrow<T>>(&self, val: V, overlap: NonZeroUsize) -> usize {
        let val = val.borrow();
        if self.not_inside(val) {
            let num_bins_overlap = 1usize.max(self.bin_count() / overlap.get());
            let dist = 
            if *val < self.first_border() { 
                self.first_border() - *val
            } else {
                *val - self.get_right()
            };
            1 + dist.to_usize().unwrap() / num_bins_overlap
        } else {
            0
        }
    }
}

impl<T> HistogramPartition for HistogramInt<T> 
where T: Clone + std::fmt::Debug
{
    fn overlapping_partition(&self, n: NonZeroUsize, overlap: usize) -> Result<Vec<Self>, HistErrors>
    {
        let mut result = Vec::with_capacity(n.get());
        let size = self.bin_count() - 1;
        let denominator = n.get() + overlap;

        for c in 0..n.get() {
            let left_index = c.checked_mul(size)
                .ok_or(HistErrors::Overflow)?
                / denominator;
            
            let zaehler = c + overlap + 1;
            let right_index = 1 + zaehler.checked_mul(size)
                .ok_or(HistErrors::Overflow)?
                / denominator;

            if left_index >= right_index {
                return Err(HistErrors::IntervalWidthZero);
            }
            
            let borders = self
                .borders()[left_index..=right_index]
                .to_vec();
            let hist = vec![0; borders.len() - 1];

            let res = Self{
                bin_borders: borders,
                hist
            };
            
            result.push(res);
        }
        Ok(result)
    }
}

impl<T> IntervalOrder for HistogramInt<T>
where T: Ord
{
    fn left_compare(&self, other: &Self) -> std::cmp::Ordering {
        let self_left = &self.bin_borders[0];
        let other_left = &other.bin_borders[0];
        let order = self_left.cmp(other_left);
        if order.is_eq() {
            let self_right = self.bin_borders.last().unwrap();
            let other_right = other.bin_borders.last().unwrap();
            return self_right.cmp(other_right);
        }
        order
    }
}


/// # Histogram for binning `usize` - alias for `HistogramInt<usize>`
/// * you should use `HistUsizeFast` instead, if your bins are `[left, left+1,..., right]`
pub type HistUsize = HistogramInt<usize>;
/// # Histogram for binning `u128` - alias for `HistogramInt<u128>`
/// * you should use `HistU128Fast` instead, if your bins are `[left, left+1,..., right]`
pub type HistU128 = HistogramInt<u128>;
/// # Histogram for binning `u64` - alias for `HistogramInt<u64>`
/// * you should use `HistU64Fast` instead, if your bins are `[left, left+1,..., right]`
pub type HistU64 = HistogramInt<u64>;
/// # Histogram for binning `u32` - alias for `HistogramInt<u32>`
/// * you should use `HistU32Fast` instead, if your bins are `[left, left+1,..., right]`
pub type HistU32 = HistogramInt<u32>;
/// # Histogram for binning `u16` - alias for `HistogramInt<u16>`
/// * you should use `HistU16Fast` instead, if your bins are `[left, left+1,..., right]`
pub type HistU16 = HistogramInt<u16>;
/// # Histogram for binning `u8` - alias for `HistogramInt<u8>`
/// * you should use `HistU8Fast` instead, if your bins are `[left, left+1,..., right]`
pub type HistU8 = HistogramInt<u8>;

/// # Histogram for binning `isize` - alias for `HistogramInt<isize>`
/// * you should use `HistIsizeFast` instead, if your bins are `[left, left+1,..., right]`
pub type HistIsize = HistogramInt<isize>;
/// # Histogram for binning `i128` - alias for `HistogramInt<i128>`
/// * you should use `HistI128Fast` instead, if your bins are `[left, left+1,..., right]`
pub type HistI128 = HistogramInt<i128>;
/// # Histogram for binning `i64` - alias for `HistogramInt<i64>`
/// * you should use `HistI64Fast` instead, if your bins are `[left, left+1,..., right]`
pub type HistI64 = HistogramInt<i64>;
/// # Histogram for binning `i32` - alias for `HistogramInt<i32>`
/// * you should use `HistI32Fast` instead, if your bins are `[left, left+1,..., right]`
pub type HistI32 = HistogramInt<i32>;
/// # Histogram for binning `i16` - alias for `HistogramInt<i16>`
/// * you should use `HistI16Fast` instead, if your bins are `[left, left+1,..., right]`
pub type HistI16 = HistogramInt<i16>;
/// # Histogram for binning `i8` - alias for `HistogramIntiu8>`
/// * you should use `HistI8Fast` instead, if your bins are `[left, left+1,..., right]`
pub type HistI8 = HistogramInt<i8>;

#[cfg(test)]
mod tests{
    use super::*;
    use rand::{SeedableRng, distributions::*};
    use rand_pcg::Pcg64Mcg;
    use num_traits::Bounded;
    fn hist_test_normal<T>(left: T, right: T)
    where T: num_traits::Bounded + PartialOrd + CheckedSub 
        + CheckedAdd + Zero + Ord + HasUnsignedVersion
        + One + NumCast + Copy + FromPrimitive + Bounded,
    std::ops::RangeInclusive<T>: Iterator<Item=T>,
    T::Unsigned: Bounded + HasUnsignedVersion<LeBytes=T::LeBytes, Unsigned=T::Unsigned> 
        + WrappingAdd + ToPrimitive + Sub<Output=T::Unsigned>
        + std::ops::Rem<Output=T::Unsigned> + FromPrimitive + Zero
        + std::cmp::Eq + std::ops::Div<Output=T::Unsigned>
        + Ord + std::ops::Mul<Output=T::Unsigned> + WrappingSub + Copy,
    std::ops::RangeInclusive<T::Unsigned>: Iterator<Item=T::Unsigned>,
    HistogramInt::<T>: std::fmt::Debug,
    {
        let bin_count = (to_u(right) - to_u(left)).to_usize().unwrap() + 1;
        let hist_wrapped =  HistogramInt::<T>::new_inclusive(left, right, bin_count);
       
        let mut hist = hist_wrapped.unwrap();
        assert!(hist.not_inside(T::max_value()));
        assert!(hist.not_inside(T::min_value()));
        for (id, i) in (left..=right).enumerate() {
            assert!(hist.is_inside(i));
            assert_eq!(hist.is_inside(i), !hist.not_inside(i));
            assert!(hist.get_bin_index(i).unwrap() == id);
            assert_eq!(hist.distance(i), 0.0);
            assert_eq!(hist.interval_distance_overlap(i, unsafe{NonZeroUsize::new_unchecked(2)}), 0);
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
        let borders: Vec<_> = hist.bin_enum_iter()
            .map(
                |bin|
                {
                    match bin {
                        Bin::InclusiveExclusive(left, right) => (left, right),
                        _ => unreachable!()
                    }
                }
            ).collect();
        assert_eq!(borders.len(), hist.bin_count());
        assert_eq!(
            HistogramInt::<T>::new_inclusive(left, T::max_value(), bin_count).expect_err("err"),
            HistErrors::Overflow
        );
    }


    #[test]
    fn hist_normal()
    {
        hist_test_normal(20usize, 31usize);
        hist_test_normal(-23isize, 31isize);
        hist_test_normal(-23i16, 31);
        hist_test_normal(1u8, 3u8);
        hist_test_normal(123u128, 300u128);
        hist_test_normal(-123i128, 300i128);

        hist_test_normal(i8::MIN + 1, i8::MAX - 1);
    }

    #[test]
    fn hist_index(){
        let hist = HistogramInt::<isize>::new(0, 20, 2).unwrap();
        assert_eq!(hist.borders(), &[0_isize, 10, 20]);
        for i in 0..=9
        {
            assert_eq!(hist.get_bin_index(i).unwrap(), 0);
        }
        for i in 10..20 {
            assert_eq!(hist.get_bin_index(i).unwrap(), 1);
        }
        assert!(hist.get_bin_index(20).is_err());
    }

    /// This test makes sure, that HistogramInt and HistogramFast return the same partitions,
    /// when the histograms are equivalent
    #[test]
    fn overlapping_partition_test()
    {

        let mut rng = Pcg64Mcg::seed_from_u64(2314668);
        let uni = Uniform::new_inclusive(-100, 100);
        let uni_n = Uniform::new_inclusive(1, 16);

        for overlap in 0..=5 {
            for _ in 0..100 {
                let n: usize = uni_n.sample(&mut rng);
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
                let hist_i = HistI8::new_inclusive(left, right, hist_fast.bin_count()).unwrap();
                let n_non_zero = NonZeroUsize::new(n).unwrap();
        
                let overlapping_f = hist_fast.overlapping_partition(n_non_zero, overlap);
                let overlapping_i = hist_i.overlapping_partition(n_non_zero, overlap);

                if overlapping_i.is_err() {
                    assert_eq!(overlapping_f.unwrap_err(), overlapping_i.unwrap_err());
                    continue;
                }

                let overlapping_i = overlapping_i.unwrap();
                let overlapping_f = overlapping_f.unwrap();

                let len = overlapping_i.len();

                for (index,(a, b)) in overlapping_f
                    .into_iter()
                    .zip(overlapping_i)
                    .enumerate()
                {
                    let bins_a: Vec<_> = a
                        .bin_enum_iter()
                        .map(
                            |bin|
                            {
                                match bin{
                                    Bin::SingleValued(val) => val,
                                    _ => unreachable!()
                                }
                            }
                        ).collect();

                    assert_eq!(bins_a.len(), a.hist().len());
                    
                    let bins_b: Vec<_> = b
                        .bin_enum_iter()
                        .map(
                            |bin|
                            {
                                match bin{
                                    Bin::InclusiveExclusive(left, right) => (left, right),
                                    _ => unreachable!()
                                }
                            }
                        ).collect();

                    assert_eq!(bins_b.len(), b.hist().len());

                    if bins_a.len() != bins_b.len()
                    {
                        println!("Fast: {} SLOW {}", a.bin_count(), b.bin_count());
                        dbg!(left, right, overlap);
                        dbg!(hist_i.bin_count(), hist_fast.bin_count());
                        dbg!(&bins_b, &bins_a);
                        eprintln!("index: {} of {}", index, len);
                    }

                    assert_eq!(bins_a.len(), bins_b.len());
                    assert_eq!(a.bin_count(), b.bin_count());
        
                    for (b_a, b_b) in bins_a.into_iter().zip(bins_b)
                    {
                        assert_eq!((b_a, b_a + 1), b_b);
                    }
                }
            }
        }
        
    }

    /// Check, that the range of the overlapping intervals contain the whole original interval
    #[test]
    fn overlapping_partition_test2()
    {
        let mut rng = Pcg64Mcg::seed_from_u64(231468);
        let uni = Uniform::new_inclusive(-100, 100);
        let uni_n = Uniform::new_inclusive(2, 6);
        for overlap in 0..=5 {
            for _ in 0..100 {
                let n: usize = uni_n.sample(&mut rng);
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
                let hist_i = HistI8::new_inclusive(left, right, hist_fast.bin_count()).unwrap();
                let overlapping_i = hist_i.overlapping_partition(NonZeroUsize::new(n).unwrap(), overlap)
                    .unwrap();

                assert_eq!(
                    overlapping_i.last().unwrap().borders().last(),
                    hist_i.borders().last()
                );

                assert_eq!(
                    overlapping_i.first().unwrap().borders().first(),
                    hist_i.borders().first()
                );
            }
        }
    }

    /// Check, that the range of the overlapping intervals contain the whole original interval
    /// Different binsize than the other test
    #[test]
    fn overlapping_partition_test3()
    {
        let mut rng = Pcg64Mcg::seed_from_u64(23148);
        let uni = Uniform::new_inclusive(-300, 300);
        let uni_n = Uniform::new_inclusive(2, 4);
        for binsize in 2..=7 {
            for overlap in 0..=5 {
                for _ in 0..100 {
                    let n: usize = uni_n.sample(&mut rng);
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
                            let hist_fast = HistI16Fast::new_inclusive(num_1, num_2).unwrap();
                            if hist_fast.bin_count() % binsize != 0 {
                                continue;
                            }
                            break (num_1, num_2)
                        }
                    };
                    let hist_fast = HistI16Fast::new_inclusive(left, right).unwrap();
                    let hist_i = HistI16::new_inclusive(left, right, hist_fast.bin_count() / binsize).unwrap();
                    let overlapping_i = hist_i.overlapping_partition(NonZeroUsize::new(n).unwrap(), overlap)
                        .unwrap();
                    
                    assert_eq!(
                        overlapping_i.last().unwrap().borders().last(),
                        hist_i.borders().last()
                    );
    
                    assert_eq!(
                        overlapping_i.first().unwrap().borders().first(),
                        hist_i.borders().first()
                    );
                }
            }
        }
        
    }

    #[test]
    fn bin_iter_test()
    {
        let hist = HistI16::new(0, 4, 4).unwrap();

        let mut bin_iter = hist.bin_iter();

        assert_eq!(bin_iter.next(), Some(&[0_i16, 1_i16]));
        assert_eq!(bin_iter.next(), Some(&[1_i16, 2_i16]));
        assert_eq!(bin_iter.next(), Some(&[2_i16, 3_i16]));
        assert_eq!(bin_iter.next(), Some(&[3_i16, 4_i16]));
        assert_eq!(bin_iter.next(), None);

        let hist = HistU8::new(0,8, 4).unwrap();

        let mut bin_iter = hist.bin_iter();

        assert_eq!(bin_iter.next(), Some(&[0_u8, 2_u8]));
        assert_eq!(bin_iter.next(), Some(&[2_u8, 4_u8]));
        assert_eq!(bin_iter.next(), Some(&[4_u8, 6_u8]));
        assert_eq!(bin_iter.next(), Some(&[6_u8, 8_u8]));
        assert_eq!(bin_iter.next(), None);
    }
}