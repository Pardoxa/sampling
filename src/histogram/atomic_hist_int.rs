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
        marker::PhantomData,
        borrow::*,
        ops::*,
        num::*,
        sync::atomic::*
    }
};

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};


/// # Generic Histogram for integer types
#[derive(Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct AtomicHistogramInt<T>
{
    pub(crate) bin_borders: Vec<T>,
    pub(crate) hist: Vec<AtomicUsize>,
}

impl<T> From<HistogramInt<T>> for AtomicHistogramInt<T>
{
    fn from(other: HistogramInt<T>) -> Self
    {
        let hist = other.hist
            .into_iter()
            .map(AtomicUsize::new)
            .collect();
        Self{
            hist, 
            bin_borders: other.bin_borders
        }
    }
}

impl<T> AtomicHistogramInt<T>{
    /// similar to `self.borders_clone` but does not allocate memory
    pub fn borders(&self) -> &[T]
    {
        &self.bin_borders
    }

    /// # Iterator over all the bins
    /// In AtomicHistogramInt a bin is defined by two values: The left border (inclusive)
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
    /// let hist = AtomicHistI8::new(0, 8, 4).unwrap();
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
    /// In AtomicHistogramInt a bin is defined by two values: The left border (inclusive)
    /// and the right border (exclusive)
    /// 
    /// This iterates over these values as well as the corresponding hit count (i.e. how often 
    /// the corresponding bin was hit)
    /// 
    /// # Note
    /// Since I am using atomics here it is possible that the hit-count changes during the iteration,
    /// if another thread writes to the histogram for example.
    /// 
    /// ## Item of Iterator
    /// `(&[left_border, right_border], number_of_hits)`
    /// ## Example
    /// ``` 
    /// use sampling::histogram::*;
    /// 
    /// let mut hist = AtomicHistUsize::new(0, 6, 3).unwrap();
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
    pub fn bin_hits_iter(&mut self) -> impl Iterator<Item = (&[T;2], usize)>
    {
        self.bin_iter()
            .zip(
                self.hist
                    .iter()
                    .map(|val| val.load(Ordering::Relaxed))
            )
    }

}


impl<T> AtomicHistogramInt<T>
where T: Sub<T, Output=T> + Add<T, Output=T> + Ord + One + Copy + NumCast
{
    #[inline]
    /// # Increment hit count 
    /// If `val` is inside the histogram, the corresponding bin count will be increased
    /// by 1 and the index corresponding to the bin in returned: `Ok(index)`.
    /// Otherwise an Error is returned
    /// ## Note
    /// * This is the same as [AtomicHistogramVal::count_val]
    /// * Uses atomic operations, so this is thread safe
    pub fn increment<V: Borrow<T>>(&self, val: V) -> Result<usize, HistErrors> {
        self.count_val(val)
    }

    #[inline]
    /// # Increment hit count
    /// Increments the hit count of the bin corresponding to `val`.
    /// If no bin corresponding to `val` exists, nothing happens
    /// # Note
    /// * Uses atomic operations for the counters, so this is thread safe
    pub fn increment_quiet<V: Borrow<T>>(&self, val: V)
    {
        let _ = self.increment(val);
    }
}


impl<T> AtomicHistogramInt<T>
where T: Copy{
    fn get_right(&self) -> T
    {
        self.bin_borders[self.bin_borders.len() - 1]
    }
}

/// This is basically ArrayWindows from the standard library
/// This will be replaced by a call to ArrayWindows as soon 
/// as ArrayWindows is no longer behind a feature gate
/// (see https://doc.rust-lang.org/std/slice/struct.ArrayWindows.html)
pub(crate) struct BorderWindow<'a, T: 'a>{
    slice_head: *const T,
    num: usize,
    marker: PhantomData<&'a [T;2]>
}

impl<'a, T: 'a> BorderWindow<'a, T>{
    pub(crate) fn new(slice: &'a [T]) -> Self
    {
        let num_windows = slice.len().saturating_sub(1);
        Self{
            slice_head: slice.as_ptr(),
            num: num_windows,
            marker: PhantomData
        }
    }
}

impl<'a, T> Iterator for BorderWindow<'a, T>
{
    type Item = &'a [T;2];

    #[inline]
    fn next(&mut self) -> Option<Self::Item>
    {
        if self.num == 0 {
            return None;
        }

        let ret = unsafe {
            &*self.slice_head.cast::<[T;2]>()
        };

        self.slice_head = unsafe{
            self.slice_head.add(1)
        };

        self.num -= 1;
        Some(ret)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>)
    {
        (self.num, Some(self.num))
    }

    #[inline]
    fn count(self) -> usize
    {
        self.num
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if self.num <= n {
            self.num = 0;
            return None;
        }
        // SAFETY:
        // This is safe because it's indexing into a slice guaranteed to be length > N.
        let ret = unsafe { &*self.slice_head.add(n).cast::<[T; 2]>() };
        // SAFETY: Guaranteed that there are at least n items remaining
        self.slice_head = unsafe { self.slice_head.add(n + 1) };

        self.num -= n + 1;
        Some(ret)
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.nth(self.num.checked_sub(1)?)
    }
}


impl<T> AtomicHistogramInt<T> 
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
        
        let hist = (0..bins).map(|_| AtomicUsize::new(0)).collect();
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

/// # Implements histogram
/// * anything that implements `Histogram` should also implement the trait `HistogramVal`
pub trait AtomicHistogram {
    /// # `self.hist[index] += 1`, `Err()` if `index` out of bounds
    #[inline(always)]
    fn count_index(&self, index: usize) -> Result<(), HistErrors>{
        self.count_multiple_index(index, 1)
    }

    /// # `self.hist[index] += count`, `Err()` if `index` out of bounds
    fn count_multiple_index(&self, index: usize, count: usize) -> Result<(), HistErrors>;

    /// # the created histogram
    /// Since this uses atomics, you can also write to the underlying hist yourself,
    /// if you so desire
    fn hist(&self) -> &[AtomicUsize];

    /// # How many bins the histogram contains
    #[inline(always)]
    fn bin_count(&self) -> usize
    {
        self.hist().len()
    }
    /// reset the histogram to zero
    fn reset(&mut self);

    /// # check if any bin was not hit yet
    /// * Uses SeqCst Ordering
    fn any_bin_zero(&self) -> bool
    {
        self.hist()
            .iter()
            .any(|val| val.load(Ordering::SeqCst) == 0)
    }
}

impl<T> AtomicHistogram for AtomicHistogramInt<T>
{
    #[inline]
    fn bin_count(&self) -> usize {
        self.hist.len()
    }

    #[inline]
    fn hist(&self) -> &[AtomicUsize] {
        &self.hist
    }

    #[inline]
    /// Uses SeqCst
    fn count_multiple_index(&self, index: usize, count: usize) -> Result<(), HistErrors> {
        match self.hist.get(index) {
            None => Err(HistErrors::OutsideHist),
            Some(val) => {
                val.fetch_add(count, Ordering::SeqCst);
                Ok(())
            },
        }
    }

    #[inline]
    fn reset(&mut self) {
        // compiles down to memset :)
        self.hist
            .iter_mut()
            .for_each(|h| *(h.get_mut()) = 0);
    }
}

/// * trait used for mapping values of arbitrary type `T` to bins
/// * used to create a histogram
pub trait AtomicHistogramVal<T>{
    /// convert val to the respective histogram index
    fn get_bin_index<V: Borrow<T>>(&self, val: V) -> Result<usize, HistErrors>;
    /// count val. `Ok(index)`, if inside of hist, `Err(_)` if val is invalid
    fn count_val<V: Borrow<T>>(&self, val: V) -> Result<usize, HistErrors>;
    /// # binning borders
    /// * the borders used to bin the values
    /// * any val which fullfills `self.border[i] <= val < self.border[i + 1]` 
    ///     will get index `i`.
    /// * **Note** that the last border is exclusive
    fn borders_clone(&self) -> Result<Vec<T>, HistErrors>;
    /// does a value correspond to a valid bin?
    fn is_inside<V: Borrow<T>>(&self, val: V) -> bool;
    /// opposite of `is_inside`
    fn not_inside<V: Borrow<T>>(&self, val: V) -> bool;
    /// get the left most border (inclusive)
    fn first_border(&self) -> T;

    /// # Get border on the right
    /// * Note: This border might be inclusive or exclusive
    fn last_border(&self) -> T;
    /// # calculates some sort of absolute distance to the nearest valid bin
    /// * any invalid numbers (like NAN or INFINITY) should have the highest distance possible
    /// * if a value corresponds to a valid bin, the distance should be zero
    fn distance<V: Borrow<T>>(&self, val: V) -> f64;
}

impl<T> AtomicHistogramVal<T> for AtomicHistogramInt<T>
where T: Ord + Sub<T, Output=T> + Add<T, Output=T> + One + NumCast + Copy
{


    fn count_val<V: Borrow<T>>(&self, val: V) -> Result<usize, HistErrors>
    {
        let id = self.get_bin_index(val)?;
        self.count_index(id)
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

    fn borders_clone(&self) -> Result<Vec<T>, HistErrors> {
        Ok(self.bin_borders.clone())
    }
}

impl<T> HistogramIntervalDistance<T> for AtomicHistogramInt<T> 
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

impl<T> HistogramPartition for AtomicHistogramInt<T> 
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
            
            let hist = (0..borders.len() - 1)
                .map(|_| AtomicUsize::new(0))
                .collect();

            let res = Self{
                bin_borders: borders,
                hist
            };
            
            result.push(res);
        }
        Ok(result)
    }
}

impl<T> IntervalOrder for AtomicHistogramInt<T>
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


/// # Histogram for binning `usize` - alias for `AtomicHistogramInt<usize>`
/// * you should use `HistUsizeFast` instead, if your bins are `[left, left+1,..., right]`
pub type AtomicHistUsize = AtomicHistogramInt<usize>;
/// # Histogram for binning `u128` - alias for `AtomicHistogramInt<u128>`
/// * you should use `HistU128Fast` instead, if your bins are `[left, left+1,..., right]`
pub type AtomicHistU128 = AtomicHistogramInt<u128>;
/// # Histogram for binning `u64` - alias for `AtomicHistogramInt<u64>`
/// * you should use `HistU64Fast` instead, if your bins are `[left, left+1,..., right]`
pub type AtomicHistU64 = AtomicHistogramInt<u64>;
/// # Histogram for binning `u32` - alias for `AtomicHistogramInt<u32>`
/// * you should use `HistU32Fast` instead, if your bins are `[left, left+1,..., right]`
pub type AtomicHistU32 = AtomicHistogramInt<u32>;
/// # Histogram for binning `u16` - alias for `AtomicHistogramInt<u16>`
/// * you should use `HistU16Fast` instead, if your bins are `[left, left+1,..., right]`
pub type AtomicHistU16 = AtomicHistogramInt<u16>;
/// # Histogram for binning `u8` - alias for `AtomicHistogramInt<u8>`
/// * you should use `HistU8Fast` instead, if your bins are `[left, left+1,..., right]`
pub type AtomicHistU8 = AtomicHistogramInt<u8>;

/// # Histogram for binning `isize` - alias for `AtomicHistogramInt<isize>`
/// * you should use `HistIsizeFast` instead, if your bins are `[left, left+1,..., right]`
pub type AtomicHistIsize = AtomicHistogramInt<isize>;
/// # Histogram for binning `i128` - alias for `AtomicHistogramInt<i128>`
/// * you should use `HistI128Fast` instead, if your bins are `[left, left+1,..., right]`
pub type AtomicHistI128 = AtomicHistogramInt<i128>;
/// # Histogram for binning `i64` - alias for `AtomicHistogramInt<i64>`
/// * you should use `HistI64Fast` instead, if your bins are `[left, left+1,..., right]`
pub type AtomicHistI64 = AtomicHistogramInt<i64>;
/// # Histogram for binning `i32` - alias for `AtomicHistogramInt<i32>`
/// * you should use `HistI32Fast` instead, if your bins are `[left, left+1,..., right]`
pub type AtomicHistI32 = AtomicHistogramInt<i32>;
/// # Histogram for binning `i16` - alias for `AtomicHistogramInt<i16>`
/// * you should use `HistI16Fast` instead, if your bins are `[left, left+1,..., right]`
pub type AtomicHistI16 = AtomicHistogramInt<i16>;
/// # Histogram for binning `i8` - alias for `AtomicHistogramIntiu8>`
/// * you should use `HistI8Fast` instead, if your bins are `[left, left+1,..., right]`
pub type AtomicHistI8 = AtomicHistogramInt<i8>;

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
    AtomicHistogramInt::<T>: std::fmt::Debug,
    {
        let bin_count = (to_u(right) - to_u(left)).to_usize().unwrap() + 1;
        let hist_wrapped =  AtomicHistogramInt::<T>::new_inclusive(left, right, bin_count);
       
        let hist = hist_wrapped.unwrap();
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
        assert_eq!(hist.borders_clone().unwrap().len() - 1, hist.bin_count());
        assert_eq!(
            AtomicHistogramInt::<T>::new_inclusive(left, T::max_value(), bin_count).expect_err("err"),
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
        let hist = AtomicHistogramInt::<isize>::new(0, 20, 2).unwrap();
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

    /// This test makes sure, that AtomicHistogramInt and HistogramFast return the same partitions,
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
                let n_nonzero = NonZeroUsize::new(n).unwrap();
                let hist_fast = HistI8Fast::new_inclusive(left, right).unwrap();
                let hist_i = HistI8::new_inclusive(left, right, hist_fast.bin_count()).unwrap();
        
                let overlapping_f = hist_fast.overlapping_partition(n_nonzero, overlap);
                let overlapping_i = hist_i.overlapping_partition(n_nonzero, overlap);

                if overlapping_i.is_err() {
                    assert_eq!(overlapping_f.unwrap_err(), overlapping_i.unwrap_err());
                    continue;
                }

                let overlapping_i = overlapping_i.unwrap();
                let overlapping_f = overlapping_f.unwrap();

                let len = overlapping_i.len();
        
                for (index,(a, b)) in overlapping_f.into_iter().zip(overlapping_i).enumerate()
                {
                    let bins_a: Vec<_> = a
                        .bin_enum_iter()
                        .map(
                            |bin|
                            {
                                match bin 
                                {
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
                                match bin 
                                {
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
                let n_nonzero = NonZeroUsize::new(n).unwrap();
                let hist_fast = HistI8Fast::new_inclusive(left, right).unwrap();
                let hist_i = HistI8::new_inclusive(left, right, hist_fast.bin_count()).unwrap();
                let overlapping_i = hist_i.overlapping_partition(n_nonzero, overlap).unwrap();

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
                    let n_nonzero = NonZeroUsize::new(n).unwrap();
                    let hist_fast = HistI16Fast::new_inclusive(left, right).unwrap();
                    let hist_i = HistI16::new_inclusive(left, right, hist_fast.bin_count() / binsize).unwrap();
                    let overlapping_i = hist_i.overlapping_partition(n_nonzero, overlap).unwrap();
                    
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