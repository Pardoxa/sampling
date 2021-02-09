//! For making a Metropolis simulation

mod metropolis;

pub use metropolis::*;


/// short for `Metropolis<E, R, S, Res, f64>`
pub type MetF64<E, R, S, Res> = Metropolis<E, R, S, Res, f64>;
/// short for `Metropolis<E, R, S, Res, f32>`
pub type MetF32<E, R, S, Res> = Metropolis<E, R, S, Res, f32>;

/// short for `Metropolis<E, R, S, Res, usize>`
pub type MetUsize<E, R, S, Res> = Metropolis<E, R, S, Res, usize>;
/// short for `Metropolis<E, R, S, Res, u128>`
pub type MetU128<E, R, S, Res> = Metropolis<E, R, S, Res, u128>;
/// short for `Metropolis<E, R, S, Res, u64>`
pub type MetU64<E, R, S, Res> = Metropolis<E, R, S, Res, u64>;
/// short for `Metropolis<E, R, S, Res, u32>`
pub type MetU32<E, R, S, Res> = Metropolis<E, R, S, Res, u32>;
/// short for `Metropolis<E, R, S, Res, u16>`
pub type MetU16<E, R, S, Res> = Metropolis<E, R, S, Res, u16>;
/// short for `Metropolis<E, R, S, Res, u8>`
pub type MetU8<E, R, S, Res> = Metropolis<E, R, S, Res, u8>;

/// short for `Metropolis<E, R, S, Res, isize>`
pub type MetIsize<E, R, S, Res> = Metropolis<E, R, S, Res, isize>;
/// short for `Metropolis<E, R, S, Res, i128>`
pub type MetI128<E, R, S, Res> = Metropolis<E, R, S, Res, i128>;
/// short for `Metropolis<E, R, S, Res, i64>`
pub type MetI64<E, R, S, Res> = Metropolis<E, R, S, Res, i64>;
/// short for `Metropolis<E, R, S, Res, i32>`
pub type MetI32<E, R, S, Res> = Metropolis<E, R, S, Res, i32>;
/// short for `Metropolis<E, R, S, Res, i16>`
pub type MetI16<E, R, S, Res> = Metropolis<E, R, S, Res, i16>;
/// short for `Metropolis<E, R, S, Res, i8>`
pub type MetI8<E, R, S, Res> = Metropolis<E, R, S, Res, i8>;