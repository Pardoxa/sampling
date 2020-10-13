use crate::*;
use rand::Rng;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CoinFlip {
    Head,
    Tail
}

impl CoinFlip
{
    pub fn turn(&mut self) {
        *self = match self {
            CoinFlip::Head => CoinFlip::Tail,
            CoinFlip::Tail => CoinFlip::Head
        };
    }
}

#[derive(Clone)]
pub struct CoinFlipSequence<R> {
    rng: R,
    seq: Vec<CoinFlip>,
}


impl<R> CoinFlipSequence<R>
    where R: Rng,
{
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
    pub fn head_count(&self) -> usize
    {
        self.seq.iter()
            .filter(|&item| *item == CoinFlip::Head)
            .count()
    }

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

impl<R> MarkovChain<usize, ()> for CoinFlipSequence<R>
where R: Rng
{
    fn m_step(&mut self) -> usize {
        let pos = self.rng.gen_range(0, self.seq.len());
        self.seq[pos].turn();
        pos
    }

    fn undo_step(&mut self, step: usize) -> () {
        self.seq[step].turn();
    }

    fn undo_step_quiet(&mut self, step: usize) {
        self.undo_step(step);   
    }
}