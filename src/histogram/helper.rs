use num_traits::{ops::wrapping::*, Bounded};
use std::mem;

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
 
macro_rules! has_unsigned_version {
    ($t:ty) => {
        has_unsigned_version!($t, $t);
    };
    ($t:ty, $u:ty) => {
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
    };
    ($($t:ty) +) => {
        
        $(has_unsigned_version!($t);)*
        
    }
        
}
has_unsigned_version!(u8 u16 u32 u64 u128 usize);

has_unsigned_version!(i8, u8);
has_unsigned_version!(i16, u16);
has_unsigned_version!(i32, u32);
has_unsigned_version!(i64, u64);
has_unsigned_version!(i128, u128);
has_unsigned_version!(isize, usize);

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