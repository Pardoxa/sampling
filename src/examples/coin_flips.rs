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
/// Result of markov Step
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
    /// You can ignore everything after here, it is just used for testing
    steps: usize,
    rejected: usize,
    accepted: usize,
    undo_count: usize
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
            steps:0,
            rejected: 0,
            accepted: 0,
            undo_count: 0
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
        let pos = self.rng.gen_range(0..self.seq.len());
        let previouse = self.seq[pos];
        // flip coin at that position
        self.seq[pos].turn();
        // information to restore the previouse state
        CoinFlipMove{
            previouse,
            index: pos
        }
    }

    /// # Only implemented for testcases
    /// Default implementation would suffice
    #[inline]
    fn m_steps(&mut self, count: usize, steps: &mut Vec<CoinFlipMove>) {
        self.steps += 1;
        steps.clear();
        steps.extend((0..count)
            .map(|_| self.m_step())
        );
    }

    /// # Only implemented for testcases
    /// Default implementation would suffice
    #[inline]
    fn m_steps_acc<Acc, AccFn>
    (
        &mut self,
        count: usize,
        steps: &mut Vec<CoinFlipMove>,
        acc: &mut Acc,
        mut acc_fn: AccFn
    )
    where AccFn: FnMut(&Self, &CoinFlipMove, &mut Acc)
    {
        self.steps += 1;
        steps.clear();
        steps.extend(
            (0..count)
                .map(|_| self.m_step_acc(acc, &mut acc_fn))
        );
    }

    fn undo_step(&mut self, step: &CoinFlipMove) {
        self.seq[step.index] = step.previouse;
    }

    #[inline]
    fn undo_step_quiet(&mut self, step: &CoinFlipMove) {
        self.undo_step(step);   
    }

    /// # Only implemented for testcases
    /// Default implementation would suffice
    fn undo_steps(&mut self, steps: &[CoinFlipMove], res: &mut Vec<()>) {
        self.undo_count += 1;
        res.clear();
        res.extend(
            steps.iter()
                .rev()
                .map(|step| self.undo_step(step))
        );
        assert_eq!(self.rejected, self.undo_count);
    }

    /// # Only implemented for testcases
    /// Default implementation would suffice
    fn undo_steps_quiet(&mut self, steps: &[CoinFlipMove]) {
        self.undo_count += 1;
        steps.iter()
            .rev()
            .for_each( |step| self.undo_step_quiet(step));
        assert_eq!(self.rejected, self.undo_count);
    }

    /// # Only implemented for testcases
    /// Default implementation would suffice
    fn steps_accepted(&mut self, _steps: &[CoinFlipMove])
    {
        self.accepted += 1;
        if self.accepted + self.rejected != self.steps{
            panic!("{} {} {}", self.steps, self.rejected, self.accepted)
        }
    }

    /// # Only implemented for testcases
    /// Default implementation would suffice
    fn steps_rejected(&mut self, _steps: &[CoinFlipMove])
    {
        self.rejected += 1;
        if self.accepted + self.rejected != self.steps{
            panic!("{} {} {}", self.steps, self.rejected, self.accepted)
        }

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

#[cfg(test)]
#[cfg(feature="replica_exchange")]
mod tests{
    use super::*;
    use rand::SeedableRng;
    use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
    use statrs::distribution::{Binomial, Discrete};
    use rand_pcg::{Pcg64Mcg, Pcg64};
    use std::num::*;

    #[test]
    fn combine()
    {
        let n = 30;
        let intervals = 3;

        let hist1 = HistUsizeFast::new_inclusive(0, 2 * n / 3)
            .unwrap();

        let hist_list1 = hist1.overlapping_partition(intervals, 1).unwrap();

        let mut rng = Pcg64Mcg::seed_from_u64(384789);

        let ensemble1 = CoinFlipSequence::new
        (
            n,
            Pcg64::from_rng(&mut rng).unwrap()
        );

        let rewl_builder1 = RewlBuilder::from_ensemble(
            ensemble1,
            hist_list1,
            1,
            NonZeroUsize::new(1999).unwrap(),
            NonZeroUsize::new(2).unwrap(),
            0.000001
        ).unwrap();

        let rewl1 = rewl_builder1.greedy_build(|e| Some(e.head_count()));

        let ensemble2 = CoinFlipSequence::new(
            n,
            Pcg64::from_rng(rng).unwrap()
        );

        let hist2 = HistUsizeFast::new_inclusive(n / 3, n)
            .unwrap();

        let hist_list2 = hist2.overlapping_partition(intervals, 1).unwrap();


        let rewl_builder2 = RewlBuilder::from_ensemble(
            ensemble2,
            hist_list2,
            1,
            NonZeroUsize::new(1999).unwrap(),
            NonZeroUsize::new(2).unwrap(),
            0.0000022
        ).unwrap();

        let mut rewl2 = rewl_builder2.greedy_build(|e| Some(e.head_count()));

        rewl2.simulate_until_convergence(|e| Some(e.head_count()));

        let mut rewl_slice = vec![rewl1, rewl2];
        rewl_slice.par_iter_mut()
            .for_each(|rewl| rewl.simulate_until_convergence(|e| Some(e.head_count())));
        

        rewl_slice.iter()
            .for_each(
                |r| 
                    {
                        r.walkers().iter()
                            .for_each(|w| println!("rewl replica_frac {}", w.replica_exchange_frac()));
                    }
                );

        let steps: u64 = rewl_slice
                .iter()
                .flat_map(|r| 
                    r.walkers()
                        .iter()
                        .map(|w| w.step_count())
                ).sum();
        println!("Ges steps rewl {}", steps);

        let binomial = Binomial::new(0.5, n as u64).unwrap();

        let prob = rewl::merged_log_prob(&rewl_slice)
            .unwrap();
        let ln_prob_true: Vec<_> = (0..=n)
             .map(|k| binomial.ln_pmf(k as u64))
             .collect();
        
        let mut rees_slice: Vec<_> = rewl_slice.into_iter()
            .map(|r| r.into_rees())
            .collect();

        rees_slice
            .par_iter_mut() 
            .for_each(
                |rees| 
                rees.simulate_until_convergence(
                    |e| Some(e.head_count()), 
                    |_,_, _|{}
                )
            );
        
        let steps: u64 = rees_slice
            .iter()
            .flat_map(|r| 
                r.walkers()
                    .iter()
                    .map(|w| w.step_count())
            ).sum();
        println!("Ges steps rees {}", steps);

        let prob_rees = rees::merged_log_prob_rees(&rees_slice).unwrap();

        let mut max_ln_difference_rewl = f64::NEG_INFINITY;
        let mut max_difference_rewl = f64::NEG_INFINITY;
        let mut frac_difference_max_rewl = f64::NEG_INFINITY;
        let mut frac_difference_min_rewl = f64::INFINITY;
        let mut average_p_dif_rewl = 0.0;

        let mut max_ln_difference_rees = f64::NEG_INFINITY;
        let mut max_difference_rees = f64::NEG_INFINITY;
        let mut frac_difference_max_rees = f64::NEG_INFINITY;
        let mut frac_difference_min_rees = f64::INFINITY;

        let iter = prob.0.
            into_iter()
            .zip(prob_rees.0)
            .zip(ln_prob_true)
            .enumerate();

        for (index, ((val_sim1, val_sim2), val_true)) in iter
        {
            println!("{} {} {} {}", index, val_sim1, val_sim2, val_true);

            let val_real = val_true.exp();
            let val_simulation1 = val_sim1.exp();
            let val_simulation2 = val_sim2.exp();

            max_difference_rewl = f64::max((val_simulation1 - val_real).abs(), max_difference_rewl);
            max_ln_difference_rewl = f64::max(max_ln_difference_rewl, (val_sim1-val_true).abs());

            let frac = val_simulation1 / val_real;
            average_p_dif_rewl += (frac - 1.0).abs();
            frac_difference_max_rewl = frac_difference_max_rewl.max(frac);
            frac_difference_min_rewl = frac_difference_min_rewl.min(frac);

            max_difference_rees = f64::max((val_simulation2 - val_real).abs(), max_difference_rees);
            max_ln_difference_rees = f64::max(max_ln_difference_rees, (val_sim2-val_true).abs());

            let frac = val_simulation2 / val_real;
            frac_difference_max_rees = frac_difference_max_rees.max(frac);
            frac_difference_min_rees = frac_difference_min_rees.min(frac);
        }
        average_p_dif_rewl /= (n+1) as f64;

        println!("Average_p_dif {}", average_p_dif_rewl);

        println!("max_ln_difference rewl: {}", max_ln_difference_rewl);
        println!("max absolute difference rewl: {}", max_difference_rewl);
        println!("max frac rewl: {}", frac_difference_max_rewl);
        println!("min frac rewl: {}", frac_difference_min_rewl);

        println!("max_ln_difference rees: {}", max_ln_difference_rees);
        println!("max absolute difference rees: {}", max_difference_rees);
        println!("max frac rees: {}", frac_difference_max_rees);
        println!("min frac rees: {}", frac_difference_min_rees);

        assert!(max_difference_rewl < 0.0006);
        assert!(max_difference_rees < 0.0005);
        assert!(frac_difference_max_rewl - 1.0 < 0.02);
        assert!(frac_difference_max_rees - 1.0 < 0.02);
        assert!((frac_difference_min_rewl - 1.0).abs() < 0.018);
        assert!((frac_difference_min_rees - 1.0).abs() < 0.012);
    }

}