use std::{
    num::NonZeroUsize, 
    ops::{
        RangeInclusive,
        Deref,
    }
};

#[derive(Clone)]
pub struct GridRangeF64{
    range: RangeInclusive<f64>,
    steps_m1: NonZeroUsize,
}

impl GridRangeF64{
    /// # This constructs the type in a native way. It cannot panic
    /// `start`: Where the range should start
    /// `end`: Where the range should end
    /// `step_m1` number of items the iterator will return minus 1. i.e., the iterator will return `steps_m1 + 1` items
    /// ## Note
    /// if either `start` or `end` are NaN, all items returned by the iterator will be NaN
    /// 
    /// If either value is infinite, the returned values will be either infity or NaN or a mix thereof
    pub fn new_native(
        start: f64,
        end: f64, 
        steps_m1: NonZeroUsize
    ) -> Self
    {
        Self{
            range: RangeInclusive::new(start, end),
            steps_m1,
        }
    }

    /// # This constructs the type in a native way. It cannot panic
    /// `start`: Where the range should start
    /// `end`: Where the range should end
    /// `steps` number of steps the iterator will return.
    /// ## Note
    /// This will panic if `steps < 2`
    pub fn new(
        start: f64,
        end: f64,
        steps: usize
    ) -> Self
    {
        assert!(steps >= 2, "steps has to be >= 2");
        let steps_m1 = NonZeroUsize::new(steps - 1).unwrap();
        Self::new_native(start, end, steps_m1)
    }

    pub fn iter(&self) -> GridRangeIterF64
    {
        GridRangeIterF64{
            start: *self.range.start(),
            end: *self.range.end(),
            steps_m1: self.steps_m1,
            exhausted: false,
            current: 0
        }
    }
}

impl Deref for GridRangeF64
{
    type Target = RangeInclusive<f64>;

    fn deref(&self) -> &Self::Target {
        &self.range
    }
}

#[derive(Clone)]
pub struct GridRangeIterF64{
    start: f64,
    end: f64,
    steps_m1: NonZeroUsize,
    current: usize,
    exhausted: bool
}



impl Iterator for GridRangeIterF64 
{
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        let next = self.start + (self.end - self.start) * (self.current as f64 / self.steps_m1.get() as f64);
        self.exhausted = self.current == self.steps_m1.get();
        if !self.exhausted{
            self.current += 1;
        }
        Some(next)
    }
    
    fn last(self) -> Option<Self::Item>
    {
        Some(self.end)
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item>
    {
        if self.exhausted{
            return None;
        }
        self.current = match self.current.checked_add(n){
            Some(v) => v,
            None => {
                self.exhausted = true;
                return None;
            }
        };
        if self.current <= self.steps_m1.get()
        {
            let next = self.start + (self.end - self.start) * (self.current as f64 / self.steps_m1.get() as f64);
            self.exhausted = self.current == self.steps_m1.get();
            if !self.exhausted{
                self.current += 1;
            }
            Some(next)
        } else {
            self.exhausted = true;
            None
        }
    }
}

#[cfg(test)]
mod grid_range_tests
{
    use super::*;
    use rand_pcg::Pcg64Mcg;
    use rand::{SeedableRng, distributions::*};

    #[test]
    fn range_test(){
        let range = GridRangeF64::new_native(0.0, 1.0, NonZeroUsize::new(2).unwrap());
        let res: Vec<f64> = range.iter().collect();

        assert_eq!(res, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn random_range_tests()
    {
        let mut rng = Pcg64Mcg::seed_from_u64(231242747);
        let rng2 = Pcg64Mcg::from_rng(&mut rng).unwrap(); 
        let dist_f64 = Uniform::new_inclusive(f64::MIN, f64::MAX);
        let dist_usize = Uniform::new_inclusive(1, usize::MAX);
        let mut f64_iter = dist_f64.sample_iter(rng);
        let mut usize_iter = dist_usize.sample_iter(rng2);

        for _ in 0..1000 {
            let left = f64_iter.next().unwrap();
            let right = f64_iter.next().unwrap();
            let steps_m1 = usize_iter.next().unwrap();
            let steps_m1 = NonZeroUsize::new(steps_m1).unwrap();

            let range = GridRangeF64::new_native(left, right, steps_m1);
            let mut iter = range.iter();
            let start = iter.next().unwrap();
            assert_eq!(start, left);
            let last = iter.last().unwrap();
            assert_eq!(last, right);
        }
    }

    #[test]
    fn nth_test()
    {
        let range = GridRangeF64::new_native(0.0, 23.128273958759, NonZeroUsize::new(123321).unwrap());
        let mut iter = range.iter();

        let nth = iter.nth(231).unwrap();
        assert_eq!(nth, 0.04332296433270351);
        let nth = iter.nth(usize::MAX);
        assert_eq!(nth, None)
    }

    #[test]
    fn nan_test()
    {
        let range = GridRangeF64::new_native(f64::NAN, 0.0, NonZeroUsize::new(23).unwrap());
        let mut iter = range.iter();

        assert!(
            iter.all(|v| v.is_nan())
        );

        let range = GridRangeF64::new_native(0.0, f64::NAN, NonZeroUsize::new(23).unwrap());
        let mut iter = range.iter();

        assert!(
            iter.all(|v| v.is_nan())
        )
    }

    #[test]
    fn inf_test()
    {
        let range = GridRangeF64::new_native(f64::INFINITY, 0.0, NonZeroUsize::new(23).unwrap());
        let mut iter = range.iter();

        assert!(
            iter.all(|v| v.is_nan())
        );

        let range = GridRangeF64::new_native(f64::NEG_INFINITY, 0.0, NonZeroUsize::new(23).unwrap());
        let mut iter = range.iter();

        assert!(
            iter.all(|v| v.is_nan())
        );

        let range = GridRangeF64::new_native(f64::NEG_INFINITY, f64::NEG_INFINITY, NonZeroUsize::new(23).unwrap());
        let mut iter = range.iter();

        assert!(
            iter.all(|v| v.is_nan())
        );

        let range = GridRangeF64::new_native(0.0, f64::NEG_INFINITY, NonZeroUsize::new(23).unwrap());
        let mut iter = range.iter();
        

        assert!(
            iter.all(|v| !v.is_finite())
        );
        
    }
}