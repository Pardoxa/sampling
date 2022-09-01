use{
    crate::histogram::*,
    std::{borrow::*, num::*},
    num_traits::{float::*, cast::*, identities::*}
};

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

/// Generic Histogram struct
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct HistogramFloat<T>
{
    bin_borders: Vec<T>,
    hist: Vec<usize>,
}

impl<T> HistogramFloat<T>{
    /// similar to `self.borders_clone` but does not allocate memory
    pub fn borders(&self) -> &Vec<T>
    {
        &self.bin_borders
    }
}

impl<T> HistogramFloat<T>
where T: Copy {
    fn get_right(&self) -> T
    {
        self.bin_borders[self.bin_borders.len() - 1]
    }
}

impl<T> HistogramFloat<T> 
where T: Float + PartialOrd + FromPrimitive {
    /// # Create a new Historgram
    /// * right exclusive, left inclusive
    /// * if you want `right` to behave (almost) the same as an inclusive border,
    /// consider using `new(left, right + T::EPSILON, bins)` (make sure, that adding Epsilon actually changes the value!)
    pub fn new(left: T, right: T, bins: usize) -> Result<Self, HistErrors>
    {
        if left >= right {
            return Err(HistErrors::IntervalWidthZero);
        }
        else if bins < 1 {
            return Err(HistErrors::NoBins);
        }
        if !left.is_finite() || !right.is_finite() {
            return Err(HistErrors::InvalidVal);
        }

        let bins_as_t = match T::from_usize(bins) {
            Some(val) => val,
            None => return Err(HistErrors::UsizeCastError),
        };

        let bin_size = (right - left) / bins_as_t;
        let hist = vec![0; bins];
        let mut bin_borders = Vec::with_capacity(bins + 1);
        bin_borders.extend((0..bins)
            .map(|val| bin_size.mul_add(T::from_usize(val).unwrap(), left)) 
        );
        bin_borders.push(right);
        Ok(
            Self{
                bin_borders,
                hist
            }
        )
    }

    /// Returns the length of the interval
    pub fn interval_length(&self) -> T
    {
        self.get_right() - self.first_border()
    }

    /// # Iterator over all the bins
    /// In HistogramFloat a bin is defined by two values: The left border (inclusive)
    /// and the right border (exclusive)
    /// 
    /// Here you get an iterator which iterates over said borders.
    /// The Iterator returns a borrowed Array of length two, where the first value is the left (inclusive) border 
    /// and the second value is the right (exclusive) border
    /// ## Example
    /// ```
    /// use sampling::histogram::*;
    /// 
    /// let hist = HistogramFloat::<f32>::new(0.0, 1.0, 2).unwrap();
    /// let mut iter = hist.bin_iter();
    /// assert_eq!(iter.next(), Some(&[0.0, 0.5]));
    /// assert_eq!(iter.next(), Some(&[0.5, 1.0]));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn bin_iter(&self) -> impl Iterator<Item = &[T;2]>
    {
        BorderWindow::new(self.bin_borders.as_slice())
    }

    /// # Iterate over all bins
    /// In HistogramFloat a bin is defined by two values: The left border (inclusive)
    /// and the right border (exclusive)
    /// 
    /// This Iterator iterates over these values as well as the corresponding hit count (
    /// i.e., how often a bin was hit)
    /// ## Item of Iterator
    /// (&[left_border, right_border], number_of_hits)
    /// ## Example
    /// ```
    /// use sampling::histogram::*;
    /// 
    /// let mut hist = HistogramFloat::<f64>::new(0.0, 1.0, 2).unwrap();
    /// 
    /// hist.increment_quiet(0.5);
    /// hist.increment_quiet(0.71253782387);
    /// 
    /// let mut iter = hist.bin_hits_iter();
    /// assert_eq!(iter.next(), Some((&[0.0, 0.5], 0)));
    /// assert_eq!(iter.next(), Some((&[0.5, 1.0], 2)));
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

    #[inline]
    /// # Increment hit count of bin
    /// This will increment the hit count of the bin corresponding to the value `val`.
    /// If the bin was valid it will return the index of the corresponding bin
    pub fn increment<B: Borrow<T>>(&mut self, val: B)-> Result<usize, HistErrors>
    {
        self.count_val(val)
    }

    #[inline]
    /// # Increment hit count
    /// Increments the hit count of the bin corresponding to `val`.
    /// If no bin corresponding to `val` exists, nothing happens
    pub fn increment_quiet<B: Borrow<T>>(&mut self, val: B)
    {
        let _ = self.increment(val);
    }
}

impl<T> Histogram for HistogramFloat<T>
{
    #[inline(always)]
    fn bin_count(&self) -> usize {
        self.hist.len()
    }

    #[inline(always)]
    fn hist(&self) -> &Vec<usize> {
        &self.hist
    }

    #[inline]
    fn count_multiple_index(&mut self, index: usize, count: usize) -> Result<(), HistErrors> {
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
        // compiles to memset ^__^
        self.hist
            .iter_mut()
            .for_each(|h| *h = 0);
    }


}

impl<T> HistogramVal<T> for HistogramFloat<T>
where T: Float + Zero + NumCast{

    fn count_val<V: Borrow<T>>(&mut self, val: V) -> Result<usize, HistErrors>
    {
        let id = self.get_bin_index(val)?;
        self.count_index(id)
            .map(|_| id)
    }

    fn distance<V: Borrow<T>>(&self, val: V) -> f64 {
        let val = val.borrow();
        if self.is_inside(val) {
            0.0
        } else if !val.is_finite() {
            f64::INFINITY
        } else if *val < self.first_border() {
            (self.first_border() - *val).to_f64().unwrap()
        } else {
            (*val - self.get_right() + T::epsilon())
                .to_f64()
                .unwrap()
        }
    }

    #[inline]
    fn first_border(&self) -> T {
        self.bin_borders[0]
    }

    #[inline]
    fn second_last_border(&self) -> T {
        self.bin_borders[self.bin_borders.len() - 2]
    }

    fn is_inside<V: Borrow<T>>(&self, val: V) -> bool {
        *val.borrow() >= self.bin_borders[0] 
            && *val.borrow() < self.bin_borders[self.bin_borders.len() - 1]
    }

    fn not_inside<V: Borrow<T>>(&self, val: V) -> bool {
        !(*val.borrow()).is_finite() 
            || *val.borrow() < self.bin_borders[0] 
            || *val.borrow() >= self.bin_borders[self.bin_borders.len() - 1]
    }


    fn get_bin_index<V: Borrow<T>>(&self, val: V) -> Result<usize, HistErrors>
    {
        let val = val.borrow();
        if !val.is_finite(){
            Err(HistErrors::InvalidVal)
        } 
        else if self.is_inside(val)
        {
            let search_res = self.bin_borders.binary_search_by(
                |v|
                v.partial_cmp(val).expect("Should never be NaN")
            );
            match search_res
            {
                Result::Ok(index) => {
                    Ok(index)
                },
                Result::Err(index_p1) => {
                    Ok(index_p1 - 1)
                }
            }
        }
        else {
            Err(HistErrors::OutsideHist)
        } 
    }

    /// consider using `self.borders()`
    fn borders_clone(&self) -> Result<Vec<T>, HistErrors> {
        Ok(self.bin_borders.clone())
    }
}

impl<T> HistogramIntervalDistance<T> for HistogramFloat<T> 
where T: Float + FromPrimitive + Zero + NumCast
{
    fn interval_distance_overlap<V: Borrow<T>>(&self, val: V, overlap: NonZeroUsize) -> usize {
        let val = val.borrow();
        
        debug_assert!(self.interval_length() > T::zero());
        debug_assert!(val.is_finite());
        if self.not_inside(val) {
            let num_bins_overlap = self.bin_count() / overlap.get();
            let dist = 
            if *val < self.first_border() { 
                let tmp = self.first_border() - *val;
                (tmp / self.interval_length()).floor()
            } else {
                let tmp = *val - self.get_right();
                (tmp / self.interval_length()).ceil()
            };
            let int_dist = dist.to_usize().unwrap();
            1 + int_dist / num_bins_overlap
        } else {
            0
        }
    }
}

/// Histogram for binning `f32` - alias for `HistogramFloat<f32>`
pub type HistF32 = HistogramFloat<f32>;

/// Histogram for binning `f64` - alias for `HistogramFloat<f64>`
pub type HistF64 = HistogramFloat<f64>;


#[cfg(test)]
mod tests{
    use rand_pcg::Pcg64Mcg;
    use rand::{distributions::*, SeedableRng};
    use super::*;
    use num_traits::Bounded;
    #[test]
    fn f64_hist()
    {
        let rng = Pcg64Mcg::new(0xcafef00dd15ea5e5);
        let dist = Uniform::new(f64::EPSILON, 1.0);
        let mut iter = dist.sample_iter(rng);

        for i in 1..100 {
            let left = iter.next().unwrap();
            let right = left + iter.next().unwrap();

            let hist = HistogramFloat::<f64>::new(left, right, i).unwrap();

            assert_eq!(left, hist.first_border(), "i={}", i);
            assert_eq!(right, hist.get_right(), "i={}", i);
            assert_eq!(i+1, hist.borders().len(), "i={}", i);

        }
    }

    fn hist_test_float<T>(left: T, right: T, bin_count: usize)
    where T: Float + num_traits::Bounded + PartialOrd 
        + One + NumCast + Copy + FromPrimitive + Bounded + std::fmt::Debug
        + PartialOrd,
    {

        let hist_wrapped =  HistogramFloat::<T>::new(left, right, bin_count);
        if hist_wrapped.is_err(){
            dbg!(&hist_wrapped);
        }
        let hist = hist_wrapped.unwrap();
        assert!(hist.not_inside(T::infinity()));
        assert!(hist.not_inside(T::nan()));
        let len = hist.borders().len();
        
        for (id, border) in hist.borders()
            .iter()
            .take(len - 1)
            .enumerate()
        {
            assert!(hist.is_inside(border));
            assert_eq!(hist.is_inside(border), !hist.not_inside(border));
            assert_eq!(hist.get_bin_index(border).unwrap(), id);
        }
        
        let last_border = hist.borders()[len - 1];
        assert!(hist.not_inside(last_border));
        assert_eq!(hist.is_inside(last_border), !hist.not_inside(last_border));
        assert!(hist.get_bin_index(last_border).is_err());
        

        for (id, border) in hist.borders()
            .iter()
            .skip(1)
            .enumerate()
        {
            let mut m_epsilon = *border;
            for mut i in 1..{
                if i > 100 {
                    i = i * i;
                }
                m_epsilon = T::epsilon().mul_add(
                    T::from_isize(-i).unwrap(), 
                    *border
                );
                if m_epsilon < *border {
                    break;
                }
            }
            assert!(hist.is_inside(m_epsilon));
            assert_eq!(hist.get_bin_index(m_epsilon).unwrap(), id);
        }
       
        assert_eq!(
            HistErrors::InvalidVal,
            HistogramFloat::<T>::new(T::nan(), right, bin_count).unwrap_err()
        );
        assert_eq!(
            HistErrors::InvalidVal,
            HistogramFloat::<T>::new(left, T::nan(), bin_count).unwrap_err()
        );
        assert_eq!(
            HistErrors::InvalidVal,
            HistogramFloat::<T>::new(left, T::infinity(), bin_count).unwrap_err()
        );
        assert_eq!(
            HistErrors::InvalidVal,
            HistogramFloat::<T>::new(T::neg_infinity(), right, bin_count).unwrap_err()
        );
    }

    #[test]
    fn hist_float()
    { 
        let mut rng = Pcg64Mcg::new(0xcafef00dd15ea5e5);
        let dist = Uniform::new(1usize, 111);
        let mut iter = dist.sample_iter(
            Pcg64Mcg::from_rng(&mut rng).unwrap()
        );
        hist_test_float(20.0, 31.0, iter.next().unwrap());
        hist_test_float(-23.0f32, 31.1232f32, iter.next().unwrap());
        hist_test_float(-13.0f32, 31.4657f32, iter.next().unwrap());
        hist_test_float(1.0f64, 3f64, iter.next().unwrap());

        let dist2 = Uniform::new(0.0, 76257f64);
        for _ in 0..10 {
            let (left, right) = loop{
                let left = dist2.sample(&mut rng);
                let right = left + dist2.sample(&mut rng);
                if left.is_finite() && right.is_finite(){
                    break (left, right);
                }
            };
            hist_test_float(left, right, iter.next().unwrap());
        }
    }
}