use num_traits::{ops::wrapping::*, Bounded};
use std::mem;
use std::marker::PhantomData;

/// Helper trait for efficient calculations in other implementations
pub trait HasUnsignedVersion {
    /// which unsigned type corresponds to this type?
    type Unsigned;
    /// Type returned by `self.to_le_bytes()`. 
    /// Depends on how many bytes are needed, to represent the number
    type LeBytes;

    /// to little endian. See implementation for integers in the standard library
    fn to_le_bytes(self) -> Self::LeBytes;

    /// from little endian. See implementation for integers in the standard library
    fn from_le_bytes(bytes: Self::LeBytes) -> Self;
}

// see link for other, somewhat more general solution
// https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=579199c02611a20cdbfc9928c00befe7
macro_rules! has_unsigned_version {
    (
        $u:ty, $t:ty $(,)?
    ) => (
        impl HasUnsignedVersion for $t {
            type Unsigned = $u;
            type LeBytes = [u8; mem::size_of::<Self>()];

            #[inline(always)]
            fn to_le_bytes(self) -> Self::LeBytes {
                self.to_le_bytes()
            }

            #[inline(always)]
            fn from_le_bytes(bytes: Self::LeBytes) -> Self {
                Self::from_le_bytes(bytes)
            }
        }   
    );
    (
        $(
            ($u:ty, $i:ty)
        ),* $(,)?
    ) => (
        $(
            has_unsigned_version!($u, $u);
            has_unsigned_version!($u, $i);
        )*
    );
}

has_unsigned_version! {
    (u8, i8),
    (u16, i16),
    (u32, i32),
    (u64, i64),
    (u128, i128),
    (usize, isize),
}

#[inline(always)]
pub(crate) fn to_u<T>(v: T) -> T::Unsigned
where T: num_traits::Bounded + HasUnsignedVersion,
    T::Unsigned: num_traits::Bounded + HasUnsignedVersion<LeBytes=T::LeBytes> + WrappingAdd
{
    let u = T::Unsigned::from_le_bytes(v.to_le_bytes());
    u.wrapping_add(&T::Unsigned::from_le_bytes(T::min_value().to_le_bytes()))
}

#[inline(always)]
pub(crate) fn from_u<T, V>(u: T) -> V
where T: num_traits::Bounded + HasUnsignedVersion + WrappingSub + Bounded,
    T::Unsigned: num_traits::Bounded + HasUnsignedVersion<LeBytes=T::LeBytes> + WrappingAdd,
    V: HasUnsignedVersion<LeBytes=T::LeBytes> + Bounded
{
    let u = u.wrapping_sub(&T::from_le_bytes(V::min_value().to_le_bytes()));
    V::from_le_bytes(u.to_le_bytes())
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


#[cfg(test)]
mod tests{
    use rand_pcg::Pcg64Mcg;
    use rand::{SeedableRng, distributions::*};
    use super::*;


    #[test]
    fn convert_and_back_ord()
    {
        let rng = Pcg64Mcg::seed_from_u64(2747);
        let dist = Uniform::new_inclusive(i8::MIN, i8::MAX);
        let mut iter = dist.sample_iter(rng);

        for _ in 0..1000
        {
            let a = iter.next().unwrap();
            let b = iter.next().unwrap();
            assert_eq!(a < b, to_u(a) < to_u(b));
        }
    }
    #[test]
    fn convert_and_back_i8()
    {
        let rng = Pcg64Mcg::seed_from_u64(2747);
        let dist = Uniform::new_inclusive(i8::MIN, i8::MAX);
        let iter = dist.sample_iter(rng);

        for i in iter.take(10000)
        {
            assert_eq!(i, from_u::<_, i8>(to_u(i)));
        }
    }
    #[test]
    fn convert_and_back_i16()
    {
        let rng = Pcg64Mcg::seed_from_u64(2736746347);
        let dist = Uniform::new_inclusive(i16::MIN, i16::MAX);
        let iter = dist.sample_iter(rng);

        for i in iter.take(10000)
        {
            assert_eq!(i, from_u::<_, i16>(to_u(i)));
        }
    }

    #[test]
    fn convert_and_back_isize()
    {
        let rng = Pcg64Mcg::seed_from_u64(27367463247);
        let dist = Uniform::new_inclusive(isize::MIN, isize::MAX);
        let iter = dist.sample_iter(rng);

        for i in iter.take(10000)
        {
            assert_eq!(i, from_u::<_, isize>(to_u(i)));
        }
    }

    #[test]
    fn convert_and_back_u128()
    {
        let rng = Pcg64Mcg::seed_from_u64(273674693247);
        let dist = Uniform::new_inclusive(u128::MIN, u128::MAX);
        let iter = dist.sample_iter(rng);

        for i in iter.take(10000)
        {
            assert_eq!(i, from_u::<_, u128>(to_u(i)));
        }
    }



    #[test]
    fn convert_and_back_i128()
    {
        let rng = Pcg64Mcg::seed_from_u64(2723674693247);
        let dist = Uniform::new_inclusive(i128::MIN, i128::MAX);
        let iter = dist.sample_iter(rng);

        for i in iter.take(10000)
        {
            assert_eq!(i, from_u::<_, i128>(to_u(i)));
        }
    }
}