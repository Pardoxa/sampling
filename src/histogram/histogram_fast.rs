use{
    crate::histogram::*,
    num_traits::{
        Bounded,
        int::*,
        cast::*,
        identities::*,
        ops::{
            checked::*, 
            wrapping::*
        }
    },
    std::{
        borrow::*,
        ops::*,
        num::*
    }
};
use super::binning::BinDisplay;

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

/// # Faster version of HistogramInt for Integers
/// provided the bins should be: (left, left +1, ..., right - 1)
/// then you should use this version!
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct HistogramFast<T> {
    left: T, 
    right: T,
    hist: Vec<usize>,
}

impl<T> BinDisplay for HistogramFast<T>
where 
T: PrimInt + HasUnsignedVersion + Copy + std::fmt::Display + WrappingAdd,
T::Unsigned: Bounded + HasUnsignedVersion<LeBytes=T::LeBytes> 
    + WrappingAdd + ToPrimitive + Sub<Output=T::Unsigned>
{
    type BinEntry = T;

    fn display_bin_iter(&'_ self) -> Box<dyn Iterator<Item=Self::BinEntry> + '_>{
        Box::new(
            self.bin_iter()
        )
    }

    fn write_bin<W: std::io::Write>(entry: &Self::BinEntry, mut writer: W) -> std::io::Result<()> {
        write!(writer, "{entry}")
    }

    fn write_header<W: std::io::Write>(&self, mut writer: W) -> std::io::Result<()> {
        write!(writer, "Bin")
    }
}

impl<T> HistogramFast<T>
where T: Copy
{
    /// Get left border, inclusive
    pub fn left(&self) -> T
    {
        self.left
    }

    /// Get right border, inclusive
    pub fn right(&self) -> T
    {
        self.right
    }

    /// # Returns the range covered by the bins as a `RangeInclusive<T>`
    pub fn range_inclusive(&self) -> RangeInclusive<T>
    {
        self.left..=self.right
    }
}

impl<T> HistogramFast<T> 
    where 
    T: PrimInt + HasUnsignedVersion + WrappingAdd,
    T::Unsigned: Bounded + HasUnsignedVersion<LeBytes=T::LeBytes> 
        + WrappingAdd + ToPrimitive + Sub<Output=T::Unsigned>
{
    /// # Create a new interval
    /// * same as `Self::new_inclusive(left, right - 1)` though with checks
    /// * That makes `left` an inclusive and `right` an exclusive border
    pub fn new(left: T, right: T) -> Result<Self, HistErrors>
    {
        let right = match right.checked_sub(&T::one()){
            Some(res) => res,
            None => return Err(HistErrors::Underflow),
        };
        Self::new_inclusive(left, right)
    }

    /// # Create new histogram with inclusive borders
    /// * `Err` if `left > right`
    /// * `left` is inclusive border
    /// * `right` is inclusive border
    pub fn new_inclusive(left: T, right: T) -> Result<Self, HistErrors>
    {
        if left > right {
            Err(HistErrors::OutsideHist)
        } else {
            let left_u = to_u(left);
            let right_u = to_u(right);
            let size = match (right_u - left_u).to_usize(){
                Some(val) => match val.checked_add(1){
                    Some(val) => val,
                    None => return Err(HistErrors::Overflow),
                },
                None => return Err(HistErrors::UsizeCastError),
            };

            Ok(
                Self{
                    left,
                    right,
                    hist: vec![0; size],
                }
            )
        }
    }

    /// # Iterator over all the bins
    /// In HistogramFast is hit only when a value matches the corresponding bin exactly,
    /// which means there exists a map between bins and corresponding values that would hit them.
    /// So a bin is perfectly defined by one value. That is what we are iterating over here
    /// 
    /// This iterates over these values
    /// # Example
    /// ```
    /// use sampling::histogram::HistogramFast;
    /// 
    /// let hist = HistogramFast::<u8>::new_inclusive(2, 5).unwrap();
    /// let vec: Vec<u8> =  hist.bin_iter().collect();
    /// assert_eq!(&vec, &[2, 3, 4, 5]);
    /// ```
    pub fn bin_iter(&self) -> impl Iterator<Item=T>
    {
        HistFastIterHelper{
            current: self.left,
            right: self.right,
            invalid: false
        }

    }

    /// # Iterator over all the bins
    /// In HistogramFast is hit only when a value matches the corresponding bin exactly,
    /// which means there exists a map between bins and corresponding values that would hit them.
    /// 
    /// This iterates over these values as well as the corresponding hit count (i.e., how often the corresponding bin was hit)
    /// # Item of Iterator
    /// `(value_corresponding_to_bin, number_of_hits)`
    /// # Example
    /// ```
    /// use sampling::histogram::HistogramFast;
    /// 
    /// let mut hist = HistogramFast::<u8>::new_inclusive(2, 5).unwrap();
    /// hist.increment(4).unwrap();
    /// hist.increment(5).unwrap();
    /// hist.increment(5).unwrap();
    /// let vec: Vec<(u8, usize)> =  hist.bin_hits_iter().collect();
    /// assert_eq!(&vec, &[(2, 0), (3, 0), (4, 1), (5, 2)]);
    /// ```
    pub fn bin_hits_iter(&'_ self) -> impl Iterator<Item=(T, usize)> + '_
    {
        self.bin_iter()
            .zip(
                self.hist
                    .iter()
                    .copied()
            )
    }

    /// checks if the range of two Histograms is equal, i.e.,
    /// if they have the same bin borders
    pub fn equal_range(&self, other: &Self) -> bool
    where T: Eq
    {
        self.left.eq(&other.left) && self.right.eq(&other.right)
    }

    /// # Add other histogram to self
    /// * will fail if the ranges are not equal, i.e., if [equal_range](Self::equal_range)
    ///     returns false
    /// * Otherwise the hit count of the bins of self will be increased 
    ///     by the corresponding hit count of other. 
    /// * other will be unchanged
    #[allow(clippy::result_unit_err)]
    pub fn try_add(&mut self, other: &Self) -> Result<(), ()>
    where T: Eq
    {
        if self.equal_range(other) {
            self.hist
                .iter_mut()
                .zip(other.hist().iter())
                .for_each(|(s, o)| *s += o);
            Ok(())
        } else {
            Err(())
        }
    }

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

pub(crate) struct HistFastIterHelper<T>
{
    pub(crate) current: T,
    pub(crate) right: T,
    pub(crate) invalid: bool,
}

impl<T> Iterator for HistFastIterHelper<T>
where 
    T: PrimInt + WrappingAdd,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T>
    {
        if self.invalid {
            return None;
        }

        let next = self.current.wrapping_add(&T::one());
        let current = std::mem::replace(&mut self.current, next);
        self.invalid = current == self.right;
        Some(
            current
        )

    }
}

pub(crate) struct BinModIterHelper<T>
{
    pub(crate) current: T,
    pub(crate) right: T,
    pub(crate) step_by: T,
    pub(crate) invalid: bool,
}

impl<T> BinModIterHelper<T>
{
    pub(crate) fn new_unchecked(left: T, right: T, step_by: T) -> Self
    {
        Self{
            current: left,
            right,
            step_by,
            invalid: false
        }
    }
}

impl<T> Iterator for BinModIterHelper<T>
where 
    T: Add::<T, Output=T> 
        + Ord + Copy + WrappingAdd
        + WrappingSub
        + One,
{
    type Item = (T, T);

    #[inline]
    fn next(&mut self) -> Option<(T, T)>
    {
        if self.invalid {
            return None;
        }

        let next = self.current.wrapping_add(&self.step_by);
        let right = next.wrapping_sub(&T::one());
        self.invalid = right == self.right;
        let left = std::mem::replace(&mut self.current, next);
        Some(
            (left, right)
        )

    }
}

impl<T> HistogramPartition for HistogramFast<T> 
where T: PrimInt + CheckedSub + ToPrimitive + CheckedAdd + One + FromPrimitive
    + HasUnsignedVersion + Bounded + WrappingAdd,
T::Unsigned: Bounded + HasUnsignedVersion<LeBytes=T::LeBytes, Unsigned=T::Unsigned> 
    + WrappingAdd + ToPrimitive + Sub<Output=T::Unsigned> + FromPrimitive + WrappingSub,
{
    fn overlapping_partition(&self, n: usize, overlap: usize) -> Result<Vec<Self>, HistErrors>
    {
        let mut result = Vec::with_capacity(n);
        let size = self.bin_count() - 1;
        let denominator = n + overlap;
        for c in 0..n {
            let left_distance = c.checked_mul(size)
                .ok_or(HistErrors::Overflow)?
                / denominator;
                    
            let left = to_u(self.left) + T::Unsigned::from_usize(left_distance)
                .ok_or(HistErrors::CastError)?;
            
            let right_distance = (c + overlap + 1).checked_mul(size)
                .ok_or(HistErrors::Overflow)?
                / denominator;
            
            let right = to_u(self.left) + T::Unsigned::from_usize(right_distance)
                .ok_or(HistErrors::CastError)?;
            
            let left = from_u(left);
            let right = from_u(right);

            result.push(Self::new_inclusive(left, right)?);
            if result.last()
                .unwrap()
                .hist
                .is_empty()
            {
                return Err(HistErrors::IntervalWidthZero);
            }

        }
        Ok(result)
    }
}

/// Histogram for binning `usize`- alias for `HistogramFast<usize>`
pub type HistUsizeFast = HistogramFast<usize>;
/// Histogram for binning `u128` - alias for `HistogramFast<u128>`
pub type HistU128Fast = HistogramFast<u128>;
/// Histogram for binning `u64` - alias for `HistogramFast<u64>`
pub type HistU64Fast = HistogramFast<u64>;
/// Histogram for binning `u32` - alias for `HistogramFast<u32>`
pub type HistU32Fast = HistogramFast<u32>;
/// Histogram for binning `u16` - alias for `HistogramFast<u16>`
pub type HistU16Fast = HistogramFast<u16>;
/// Histogram for binning `u8` - alias for `HistogramFast<u8>`
pub type HistU8Fast = HistogramFast<u8>;

/// Histogram for binning `isize` - alias for `HistogramFast<isize>`
pub type HistIsizeFast = HistogramFast<isize>;
/// Histogram for binning `i128` - alias for `HistogramFast<i128>`
pub type HistI128Fast = HistogramFast<i128>;
/// Histogram for binning `i64` - alias for `HistogramFast<i64>`
pub type HistI64Fast = HistogramFast<i64>;
/// Histogram for binning `i32` - alias for `HistogramFast<i32>`
pub type HistI32Fast = HistogramFast<i32>;
/// Histogram for binning `i16` - alias for `HistogramFast<i16>`
pub type HistI16Fast = HistogramFast<i16>;
/// Histogram for binning `i8` - alias for `HistogramFastiu8>`
pub type HistI8Fast = HistogramFast<i8>;


impl<T> Histogram for HistogramFast<T> 
{

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
    fn hist(&self) -> &Vec<usize> {
        &self.hist
    }

    #[inline]
    fn bin_count(&self) -> usize {
        self.hist.len()
    }

    #[inline]
    fn reset(&mut self) {
        // compiles to memset =)
        self.hist
            .iter_mut()
            .for_each(|h| *h = 0);
    }
} 

impl<T> HistogramVal<T> for HistogramFast<T>
where T: PrimInt + HasUnsignedVersion + WrappingAdd,
    T::Unsigned: Bounded + HasUnsignedVersion<LeBytes=T::LeBytes> 
        + WrappingAdd + ToPrimitive + Sub<Output=T::Unsigned>
{
    #[inline]
    fn first_border(&self) -> T {
        self.left
    }

    fn last_border(&self) -> T {
        self.right
    }

    #[inline(always)]
    fn last_border_is_inclusive(&self) -> bool {
        true
    }

    fn distance<V: Borrow<T>>(&self, val: V) -> f64 {
        let val = val.borrow();
        if self.not_inside(val) {
            let dist = if *val < self.first_border() {
                self.first_border() - *val
            } else {
                val.saturating_sub(self.right)
            };
            dist.to_f64()
                .unwrap_or(f64::INFINITY)
        } else {
            0.0
        }
    }

    #[inline(always)]
    fn get_bin_index<V: Borrow<T>>(&self, val: V) -> Result<usize, HistErrors> {
        let val = *val.borrow();
        if val <= self.right{
            match val.checked_sub(&self.left) {
                None => {
                    let left = self.left.to_isize()
                        .ok_or(HistErrors::CastError)?;
                    let val = val.to_isize()
                        .ok_or(HistErrors::CastError)?;
                    match val.checked_sub(left){
                        None => Err(HistErrors::OutsideHist),
                        Some(index) => {
                            index.to_usize()
                                .ok_or(HistErrors::OutsideHist)
                        }
                    }
                },
                Some(index) => index.to_usize()
                    .ok_or(HistErrors::OutsideHist)
            }
        } else {
            Err(HistErrors::OutsideHist)
        }
    }

    /// # Iterator over the bins
    /// * This iterator will always return SingleValued bins
    /// * Consider using `self.bin_iter()` instead, its more efficient
    fn bin_enum_iter(&self) -> Box<dyn Iterator<Item=Bin<T>> + '_>{
        
        let iter = self
            .bin_iter()
            .map(|bin| Bin::SingleValued(bin));
        Box::new(iter)
    }

    #[inline]
    fn is_inside<V: Borrow<T>>(&self, val: V) -> bool {
        let val = *val.borrow();
        val >= self.left && val <= self.right
    }

    #[inline]
    fn not_inside<V: Borrow<T>>(&self, val: V) -> bool {
        let val = *val.borrow();
        val > self.right || val < self.left
    }

    #[inline]
    fn count_val<V: Borrow<T>>(&mut self, val: V) -> Result<usize, HistErrors> {
        let index = self.get_bin_index(val)?;
        self.hist[index] += 1;
        Ok(index)
    }
}

impl<T> HistogramIntervalDistance<T> for HistogramFast<T> 
where Self: HistogramVal<T>,
    T: PartialOrd + std::ops::Sub<Output=T> + NumCast + Copy
{
    fn interval_distance_overlap<V: Borrow<T>>(&self, val: V, overlap: NonZeroUsize) -> usize {
        let val = val.borrow();
        if self.not_inside(val) {
            let num_bins_overlap = 1usize.max(self.bin_count() / overlap.get());
            let dist = 
            if *val < self.left { 
                self.left - *val
            } else {
                *val - self.right
            };
            1 + dist.to_usize().unwrap() / num_bins_overlap
        } else {
            0
        }
    }
}

impl<T> HistogramCombine for HistogramFast<T>
    where   Self: HistogramVal<T>,
    T: PrimInt + HasUnsignedVersion + WrappingAdd,
    T::Unsigned: Bounded + HasUnsignedVersion<LeBytes=T::LeBytes> 
    + WrappingAdd + ToPrimitive + Sub<Output=T::Unsigned>
{
    fn encapsulating_hist<S>(hists: &[S]) -> Result<Self, HistErrors>
    where S: Borrow<Self> {
        if hists.is_empty() {
            Err(HistErrors::EmptySlice)
        } else if hists.len() == 1 {
            let h = hists[0].borrow();
            Ok(Self{
                left: h.left,
                right: h.right,
                hist: vec![0; h.hist.len()]
            })
        } else {
            let mut min = hists[0].borrow().left;
            let mut max = hists[0].borrow().right;
            hists[1..].iter()
                .for_each(
                    |h|
                    {
                        let h = h.borrow();
                        if h.left < min {
                            min = h.left;
                        }
                        if h.right > max {
                            max = h.right;
                        }
                    }
                );
            Self::new_inclusive(min, max)
        }
    }

    fn align<S>(&self, right: S) -> Result<usize, HistErrors>
    where S: Borrow<Self> {
        let right = right.borrow();

        if self.is_inside(right.left) {
            (to_u(right.left) - to_u(self.left))
                .to_usize()
                .ok_or(HistErrors::UsizeCastError)
        } else { 
            Err(HistErrors::OutsideHist)
        }
    }
}

impl<T> IntervalOrder for HistogramFast<T>
where T: PrimInt
{
    fn left_compare(&self, other: &Self) -> std::cmp::Ordering {
        let order =  self.left.cmp(&other.left);
        if order.is_eq() {
            return self.right.cmp(&other.right)
        }
        order
    }
}



#[cfg(test)]
mod tests{
    use super::*;
    use rand::{distributions::*, SeedableRng};
    use rand_pcg::Pcg64Mcg;

    fn hist_test_fast<T>(left: T, right: T)
    where T: PrimInt + num_traits::Bounded + PartialOrd + CheckedSub + One 
        + NumCast + Copy + FromPrimitive + HasUnsignedVersion + WrappingAdd,
    std::ops::RangeInclusive<T>: Iterator<Item=T>,
    T::Unsigned: Bounded + HasUnsignedVersion<LeBytes=T::LeBytes> 
        + WrappingAdd + ToPrimitive + Sub<Output=T::Unsigned>
    {
        let mut hist = HistogramFast::<T>::new_inclusive(left, right).unwrap();
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
        assert_eq!(hist.bin_enum_iter().count(), hist.bin_count());
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