use crate::*;
use rand::Rng;

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
/// # Result of flipping a coin
pub enum CoinFlip {
    /// The result is Head
    Head,
    /// The result is Tail
    Tail
}

impl CoinFlip
{
    /// Turn Coin around, i.e., invert CoinFlip
    pub fn turn(&mut self) {
        *self = match self {
            CoinFlip::Head => CoinFlip::Tail,
            CoinFlip::Tail => CoinFlip::Head
        };
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct CoinFlipMove{
    previouse: CoinFlip,
    index: usize,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
/// # A sequence of Coin flips. Contains random Number generator
pub struct CoinFlipSequence<R> {
    rng: R,
    seq: Vec<CoinFlip>,
}


impl<R> CoinFlipSequence<R>
    where R: Rng,
{
    /// Create new coin flip sequence
    /// * length `n`
    /// * use `rng` as random number generator
    pub fn new(n: usize, mut rng: R) -> Self
    {
        let mut seq = Vec::with_capacity(n);
        seq.extend(
            (0..n).map(|_| {
                if rng.gen::<bool>() {
                    CoinFlip::Tail
                } else {
                    CoinFlip::Head
                }
            })
        );
        Self{
            rng,
            seq,
        }
    }
}


impl<R> CoinFlipSequence<R>
{
    /// Count how often `Head` occurs in the Coin flip sequence
    pub fn head_count(&self) -> usize
    {
        self.seq.iter()
            .filter(|&item| *item == CoinFlip::Head)
            .count()
    }

    /// * Calculate the head count, if a previouse head count of the ensemble and the 
    /// markov steps leading to the current state are known
    /// * `head_count` is updated
    /// * might **panic** if `step` was not the markov step leading from the ensemble with `head_count`
    /// to the current ensemble - if it does not panic, the result will be wrong
    pub fn update_head_count(&self, step: &CoinFlipMove, head_count: &mut usize)
    {
        match step.previouse {
            CoinFlip::Head => {
                *head_count -= 1;
            },
            CoinFlip::Tail => {
                *head_count += 1;
            }
        }
    }

    /// Count many times `Head` occured in a row
    /// * uses maximum value, i.e., for the sequence `HHTHHHT` it will return 3
    pub fn max_heads_in_a_row(&self) -> usize
    {
        let mut current_heads = 0;
        let mut max_heads = 0;
        for flip in self.seq.iter()
        {
            match flip {
                CoinFlip::Head => current_heads += 1,
                CoinFlip::Tail => {
                    max_heads = max_heads.max(current_heads);
                    current_heads = 0;
                }
            }
        }
        max_heads.max(current_heads)
    }
}

impl<R> MarkovChain<CoinFlipMove, ()> for CoinFlipSequence<R>
where R: Rng
{
    /// Perform a markov step
    fn m_step(&mut self) -> CoinFlipMove {
        // draw a random position
        let pos = self.rng.gen_range(0,self.seq.len());
        let previouse = self.seq[pos];
        // flip coin at that position
        self.seq[pos].turn();
        // information to restore the previouse state
        CoinFlipMove{
            previouse,
            index: pos
        }
    }

    fn undo_step(&mut self, step: &CoinFlipMove) -> () {
        self.seq[step.index] = step.previouse;
    }

    #[inline]
    fn undo_step_quiet(&mut self, step: &CoinFlipMove) {
        self.undo_step(step);   
    }
}

impl<R> HasRng<R> for CoinFlipSequence<R>
    where R: Rng
{
    fn rng(&mut self) -> &mut R {
        &mut self.rng
    }

    fn swap_rng(&mut self, rng: &mut R) {
        std::mem::swap(&mut self.rng, rng);
    }
}