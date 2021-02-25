use crate::*;
use crate::{rees::replica_exchange, rewl::Rewl};
use rand::{Rng, prelude::SliceRandom};
use std::{num::NonZeroUsize, sync::*, cmp::*};
use rayon::{prelude::*};
use glue_helper::subtract_max;

#[cfg(feature = "sweep_time_optimization")]
use std::cmp::Reverse;

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

#[derive(Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct ReplicaExchangeEntropicSampling<Extra, Ensemble, R, Hist, Energy, S, Res>
{
    pub(crate) chunk_size: NonZeroUsize,
    pub(crate) ensembles: Vec<RwLock<Ensemble>>,
    pub(crate) walker: Vec<ReesWalker<R, Hist, Energy, S, Res>>,
    pub(crate) replica_exchange_mode: bool,
    pub(crate) extra: Vec<Extra>
}


pub type Rees<Extra, Ensemble, R, Hist, Energy, S, Res> = ReplicaExchangeEntropicSampling<Extra, Ensemble, R, Hist, Energy, S, Res>;

impl<Ensemble, R, Hist, Energy, S, Res> From<Rewl<Ensemble, R, Hist, Energy, S, Res>> for Rees<(), Ensemble, R, Hist, Energy, S, Res>
where Hist: Histogram
{
    fn from(rewl: Rewl<Ensemble, R, Hist, Energy, S, Res>) -> Self 
    {
        let extra = vec![(); rewl.walker.len()];

        let walker = rewl.walker
            .into_iter()
            .map(|w| w.into())
            .collect();
        
        Self{
            extra,
            chunk_size: rewl.chunk_size,
            walker,
            replica_exchange_mode: rewl.replica_exchange_mode,
            ensembles: rewl.ensembles,
        }
    }
}

impl<Extra, Ensemble, R, Hist, Energy, S, Res> Rees<Extra, Ensemble, R, Hist, Energy, S, Res>
{
    pub fn is_finished(&self) -> bool
    {
        self.walker
            .iter()
            .all(|w| w.is_finished())
    }

    /// # Get the number of intervals present
    pub fn num_intervals(&self) -> usize
    {
        self.walker.len() / self.chunk_size.get()
    }

    /// # How many walkers are there in total?
    pub fn num_walkers(&self) -> usize
    {
        self.walker.len()
    }

    /// Returns number of walkers per interval
    pub fn walkers_per_interval(&self) -> NonZeroUsize
    {
        self.chunk_size
    }

    pub fn walkers(&self) -> &Vec<ReesWalker<R, Hist, Energy, S, Res>>
    {
        &self.walker
    }

    /// # Change step size for markov chain of walkers
    /// * changes the step size used in the sweep
    /// * changes step size of all walkers in the nth interval
    /// * returns Err if index out of bounds, i.e., the requested interval does not exist
    /// * interval counting starts at 0, i.e., n=0 is the first interval
    pub fn change_step_size_of_interval(&mut self, n: usize, step_size: usize) -> Result<(), ()>
    {
        let start = n * self.chunk_size.get();
        let end = start + self.chunk_size.get();
        if self.walker.len() < end {
            Err(())
        } else {
            let slice = &mut self.walker[start..start+self.chunk_size.get()];
            slice.iter_mut()
                .for_each(|entry| entry.step_size_change(step_size));
            Ok(())
        }
    }

    /// # Change sweep size for markov chain of walkers
    /// * changes the sweep size used in the sweep
    /// * changes sweep size of all walkers in the nth interval
    /// * returns Err if index out of bounds, i.e., the requested interval does not exist
    /// * interval counting starts at 0, i.e., n=0 is the first interval
    pub fn change_sweep_size_of_interval(&mut self, n: usize, sweep_size: NonZeroUsize) -> Result<(), ()>
    {
        let start = n * self.chunk_size.get();
        let end = start + self.chunk_size.get();
        if self.walker.len() < end {
            Err(())
        } else {
            let slice = &mut self.walker[start..start+self.chunk_size.get()];
            slice.iter_mut()
                .for_each(|entry| entry.sweep_size_change(sweep_size));
            Ok(())
        }
    }

    /// # Read access to your extra information
    pub fn extra_slice(&self) -> &[Extra]
    {
        &self.extra
    }

    /// # Write access to your extra information
    pub fn extra_slice_mut(&mut self) -> &mut[Extra]
    {
        &mut self.extra
    }

    /// # Remove extra vector
    /// * returns tuple of Self (without extra, i.e., `Rees<(), Ensemble, R, Hist, Energy, S, Res>`)
    /// and vector of Extra
    pub fn unpack_extra(self) -> (Rees<(), Ensemble, R, Hist, Energy, S, Res>, Vec<Extra>)
    {
        let old_extra = self.extra;
        let extra = vec![(); old_extra.len()];
        let rees = Rees{
            extra,
            walker: self.walker,
            chunk_size: self.chunk_size,
            ensembles: self.ensembles,
            replica_exchange_mode: self.replica_exchange_mode,
        };
        (
            rees,
            old_extra
        )
    }

    /// # Swap the extra vector
    /// * Note: len of extra has to be the same as `self.num_walkers()` (which is the same as `self.extra_slice().len()`)
    /// otherwise an Err is returned
    pub fn swap_extra<Extra2>(self, new_extra: Vec<Extra2>) -> Result<(Rees<Extra2, Ensemble, R, Hist, Energy, S, Res>, Vec<Extra>), ()>
    {
        if self.extra.len() != new_extra.len(){
            Err(())
        } else {
            let old_extra = self.extra;
            let rees = Rees{
                extra: new_extra,
                walker: self.walker,
                chunk_size: self.chunk_size,
                ensembles: self.ensembles,
                replica_exchange_mode: self.replica_exchange_mode,
            };
            Ok(
                (
                    rees,
                    old_extra
                )
            )
        }
    }
}

impl<Ensemble, R, Hist, Energy, S, Res> Rees<(), Ensemble, R, Hist, Energy, S, Res>
{
    /// # Add extra information to your Replica Exchange entropic sampling simulation
    /// * can be used to, e.g., print stuff during the simulation, or write it to a file and so on
    pub fn add_extra<Extra>(self, extra: Vec<Extra>) -> Result<Rees<Extra, Ensemble, R, Hist, Energy, S, Res>, (Self, Vec<Extra>)>
    {
        if self.walker.len() != extra.len(){
            Err(
                (
                    self,
                    extra
                )
            )
        } else {
            let rees = Rees{
                extra,
                walker: self.walker,
                chunk_size: self.chunk_size,
                ensembles: self.ensembles,
                replica_exchange_mode: self.replica_exchange_mode,
            };
            Ok(
                rees
            )
        }
    }

}


impl<Extra, Ensemble, R, Hist, Energy, S, Res> Rees<Extra, Ensemble, R, Hist, Energy, S, Res>
where Ensemble: Send + Sync + MarkovChain<S, Res>,
    R: Send + Sync + Rng,
    Extra: Send + Sync,
    Hist: Send + Sync + Histogram + HistogramVal<Energy>,
    Energy: Send + Sync + Clone,
    S: Send + Sync,
    Res: Send + Sync,
{

    pub fn refine(&mut self)
    {
        self.walker
            .par_iter_mut()
            .for_each(|w| w.refine());
    }


    /// # Sweep
    /// * Performs one sweep of the Replica exchange entropic sampling simulation
    /// * You can make a complete simulation, by repeatatly calling this method
    /// until `self.is_finished()` returns true
    pub fn sweep<F, P>
    (
        &mut self,
        energy_fn: F,
        extra_fn: P
    )
    where F: Fn(&mut Ensemble) -> Option<Energy> + Copy + Send + Sync,
        P: Fn(&ReesWalker<R, Hist, Energy, S, Res>, &mut Ensemble, &mut Extra) -> () + Copy + Send + Sync,
    {
        let slice = self.ensembles.as_slice();

        #[cfg(not(feature = "sweep_time_optimization"))]
        let walker = &mut self.walker;

        #[cfg(feature = "sweep_time_optimization")]
        let mut walker = 
        {
            let mut walker = Vec::with_capacity(self.walker.len());
            walker.extend(
                self.walker.iter_mut()
            );
            walker.par_sort_unstable_by_key(|w| Reverse(w.duration()));
            walker
        };


        walker
            .par_iter_mut()
            .zip(self.extra.par_iter_mut())
            .for_each(|(w, extra)| w.sweep(slice, extra, extra_fn, energy_fn));

        // replica exchange
        if self.walkers_per_interval().get() > 1 {
            let exchange_m = self.replica_exchange_mode;
        
            self.walker
            .par_chunks_mut(self.chunk_size.get())
            .for_each(
                |chunk|
                {
                    let mut shuf = Vec::with_capacity(chunk.len());
                    if let Some((first, rest)) = chunk.split_first_mut(){
                        shuf.extend(
                            rest.iter_mut()
                        );
                        shuf.shuffle(&mut first.rng);
                        shuf.push(first);
                        let s = if exchange_m {
                            &mut shuf
                        } else {
                            &mut shuf[1..]
                        };
                        
                        s.chunks_exact_mut(2)
                            .for_each(
                                |c|
                                {
                                    let ptr = c.as_mut_ptr();
                                    unsafe{
                                        let a = &mut *ptr;
                                        let b = &mut *ptr.offset(1);
                                        replica_exchange(a, b);
                                    }
                                }
                            );
                    }
                }
            );
        }

        let walker_slice = if self.replica_exchange_mode 
        {
            &mut self.walker
        } else {
            &mut self.walker[self.chunk_size.get()..]
        };
        self.replica_exchange_mode = !self.replica_exchange_mode;

        let chunk_size = self.chunk_size;

        walker_slice
            .par_chunks_exact_mut(2 * self.chunk_size.get())
            .for_each(
                |walker_chunk|
                {
                    let (slice_a, slice_b) = walker_chunk.split_at_mut(chunk_size.get());
                    
                    let mut slice_b_shuffled: Vec<_> = slice_b.iter_mut().collect();
                    slice_b_shuffled.shuffle(&mut slice_a[0].rng);

                    for (walker_a, walker_b) in slice_a.iter_mut()
                        .zip(slice_b_shuffled.into_iter())
                    {
                        replica_exchange(walker_a, walker_b);
                    }
                }
            )
    }

    /// # Perform the Replica exchange simulation
    /// * will simulate until **all** walkers are finished
    /// * extra_fn should be used for example for writing Data to a file 
    pub fn simulate_until_convergence<F, P>(
        &mut self,
        energy_fn: F,
        extra_fn: P
    )
    where 
        Ensemble: Send + Sync,
        R: Send + Sync,
        F: Fn(&mut Ensemble) -> Option<Energy> + Copy + Send + Sync,
        P: Fn(&ReesWalker<R, Hist, Energy, S, Res>, &mut Ensemble, &mut Extra) -> () + Copy + Send + Sync,
    {
        while !self.is_finished()
        {
            self.sweep(energy_fn, extra_fn);
        }
    }

    /// # Perform the Replica exchange simulation
    /// * will simulate until **all** walkers are finished **or**
    /// * until condition returns false
    pub fn simulate_while<F, C, P>(
        &mut self,
        energy_fn: F,
        mut condition: C,
        extra_fn: P
    )
    where 
        Ensemble: Send + Sync,
        R: Send + Sync,
        F: Fn(&mut Ensemble) -> Option<Energy> + Copy + Send + Sync,
        C: FnMut(&Self) -> bool,
        P: Fn(&ReesWalker<R, Hist, Energy, S, Res>, &mut Ensemble, &mut Extra) -> () + Copy + Send + Sync,
    {
        while !self.is_finished() && condition(&self)
        {
            self.sweep(energy_fn, extra_fn);
        }
    }


    fn merged_log_probability_helper(&self) -> Result<(Vec<usize>, Vec<usize>, Vec<Vec<f64>>, Hist), HistErrors>
    where Hist: HistogramCombine
    {
        // get the log_probabilities - the walkers over the same intervals are merged
        let mut log_prob: Vec<_> = self.walker
            .par_chunks(self.chunk_size.get())
            .map(get_merged_refined_walker_prob)
            .collect();
        
        log_prob
            .par_iter_mut()
            .for_each(|v| subtract_max(v));


        // get the derivative, for merging later
        let derivatives: Vec<_> = log_prob.par_iter()
            .map(|v| derivative_merged(v))
            .collect();

        let hists: Vec<_> = self.walker
            .iter()
            .step_by(self.chunk_size.get())
            .map(|w| w.hist())
            .collect();

        let e_hist = Hist::encapsulating_hist(&hists)?;

        let alignment  = hists.iter()
            .zip(hists.iter().skip(1))
            .map(|(&left, &right)| left.align(right))
            .collect::<Result<Vec<_>, _>>()?;
        
        let merge_points: Vec<_> = calc_merge_points(&alignment, &derivatives);

        Ok(
            (
                merge_points,
                alignment,
                log_prob,
                e_hist
            )
        )

    }

    /// # Result of the simulations!
    /// This is what we do the simulation for!
    /// 
    /// It returns the natural logarithm of the normalized (i.e. sum=1 within numerical precision) probability density and the 
    /// histogram, which contains the corresponding bins.
    ///
    /// Failes if the internal histograms (invervals) do not align. Might fail if 
    /// there is no overlap between neighboring intervals 
    pub fn merged_log_prob(&self) -> Result<(Hist, Vec<f64>), HistErrors>
    where Hist: HistogramCombine
    {
        let (mut log_prob, e_hist) = self.merged_log_probability()?;

        norm_ln_prob(&mut log_prob);
        
        Ok((e_hist, log_prob))
    }

    /// # Results of the simulation
    /// 
    /// This is what we do the simulation for!
    /// 
    /// It returns histogram, which contains the corresponding bins and
    /// the natural logarithm of the normalized (i.e. sum=1 within numerical precision) 
    /// probability density. Lastly it returns the vector of the aligned probability estimates (also ln) of the
    /// different intervals. This can be used to see, how good the simulation worked,
    /// e.g., by plotting them to see, if they match
    ///
    /// ## Notes
    /// Failes if the internal histograms (invervals) do not align. Might fail if 
    /// there is no overlap between neighboring intervals 
    pub fn merged_log_prob_and_aligned(&self) -> Result<(Hist, Vec<f64>, Vec<Vec<f64>>), HistErrors>
    where Hist: HistogramCombine 
    {
        let (e_hist, mut log_prob, mut aligned) = self.merged_log_probability_and_align()?;
        
        let shift = norm_ln_prob(&mut log_prob);
        
        aligned.par_iter_mut()
            .for_each(
                |aligned|
                {
                    aligned.iter_mut()
                        .for_each(|val| *val -= shift)
                }
            );
        Ok(
            (
                e_hist,
                log_prob,
                aligned
            )
        )
    }

    fn merged_log_probability(&self) -> Result<(Vec<f64>, Hist), HistErrors>
    where Hist: HistogramCombine
    {
        let (merge_points, alignment, log_prob, e_hist) = self.merged_log_probability_helper()?;
        Ok(
            only_merged(
                merge_points,
                alignment,
                log_prob,
                e_hist
            )
        )
    }

    fn merged_log_probability_and_align(&self) -> Result<(Hist, Vec<f64>, Vec<Vec<f64>>), HistErrors>
    where Hist: HistogramCombine
    {
        let (merge_points, alignment, log_prob, e_hist) = self.merged_log_probability_helper()?;
        merged_and_aligned(
            self.walker.iter()
                    .step_by(self.walkers_per_interval().get())
                    .map(|v| v.hist()),
            merge_points,
            alignment,
            log_prob,
            e_hist
        )
    }
}

pub fn merged_log_prob<Extra, Ensemble, R, Hist, Energy, S, Res>(rees: &[Rees<Extra, Ensemble, R, Hist, Energy, S, Res>]) -> Result<(Vec<f64>, Hist), HistErrors>
where Hist: HistogramVal<Energy> + HistogramCombine + Send + Sync,
    Energy: PartialOrd
{
    let merged_prob = merged_probs(rees);
    let container = combine_container(rees, &merged_prob);
    let (merge_points, alignment, log_prob, e_hist) = align(&container)?;
    Ok(
        only_merged(
            merge_points,
            alignment,
            log_prob,
            e_hist
        )
    )
}

pub fn merged_log10_prob<Extra, Ensemble, R, Hist, Energy, S, Res>(rees: &[Rees<Extra, Ensemble, R, Hist, Energy, S, Res>]) -> Result<(Vec<f64>, Hist), HistErrors>
where Hist: HistogramVal<Energy> + HistogramCombine + Send + Sync,
    Energy: PartialOrd
{
    let mut res = merged_log_prob(rees)?;
    ln_to_log10(&mut res.0);
    Ok(res)
}

pub fn merged_log_probability_and_align<Ensemble, R, Hist, Energy, S, Res, Extra>(rees: &[Rees<Extra, Ensemble, R, Hist, Energy, S, Res>]) -> Result<(Hist, Vec<f64>, Vec<Vec<f64>>), HistErrors>
where Hist: HistogramCombine + HistogramVal<Energy> + Send + Sync,
    Energy: PartialOrd
{
    let merged_prob = merged_probs(rees);
    let container = combine_container(rees, &merged_prob);
    let (merge_points, alignment, log_prob, e_hist) = align(&container)?;
    merged_and_aligned(
        container.iter()
            .map(|c| c.1),
        merge_points,
        alignment,
        log_prob,
        e_hist
    )
}

pub fn merged_log10_probability_and_align<Ensemble, R, Hist, Energy, S, Res, Extra>(rees: &[Rees<Extra, Ensemble, R, Hist, Energy, S, Res>]) -> Result<(Hist, Vec<f64>, Vec<Vec<f64>>), HistErrors>
where Hist: HistogramCombine + HistogramVal<Energy> + Send + Sync,
    Energy: PartialOrd
{
    let mut res = merged_log_probability_and_align(rees)?;
    ln_to_log10(&mut res.1);
    res.2.par_iter_mut()
        .for_each(|slice| ln_to_log10(slice));
    Ok(res)
}

fn combine_container<'a, Ensemble, R, Hist, Energy, S, Res, Extra>(rees: &'a [Rees<Extra, Ensemble, R, Hist, Energy, S, Res>], merged_probs: &'a [Vec<f64>]) ->  Vec<(&'a [f64], &'a Hist)>
where Hist: HistogramVal<Energy> + HistogramCombine,
    Energy: PartialOrd
{
    let hists: Vec<_> = rees.iter()
        .flat_map(
            |r|
            {
                r.walkers()
                    .iter()
                    .step_by(r.walkers_per_interval().get())
                    .map(|w| w.hist())
            }
        ).collect();

    assert_eq!(hists.len(), merged_probs.len());

    let mut container: Vec<_> = merged_probs.iter()
        .zip(hists.into_iter())
        .map(|(prob, hist)| (prob.as_slice(), hist))
        .collect();

    container
        .sort_unstable_by(
            |a, b|
                {
                    a.1.first_border()
                        .partial_cmp(&b.1.first_border())
                        .unwrap_or(Ordering::Equal)
                }
            );
    container
}

fn merged_probs<Ensemble, R, Hist, Energy, S, Res, Extra>(rees: &[Rees<Extra, Ensemble, R, Hist, Energy, S, Res>]) -> Vec<Vec<f64>>
where Hist: Histogram
{
    let merged_probs: Vec<_> = rees.iter()
        .flat_map(
            |rees|
            {
                rees.walkers()
                    .chunks(rees.walkers_per_interval().get())
                    .map(get_merged_walker_prob)
            }
        ).collect();
    merged_probs
}

fn get_merged_walker_prob<R, Hist, Energy, S, Res>(walker: &[ReesWalker<R, Hist, Energy, S, Res>]) -> Vec<f64>
where Hist: Histogram
{
    let log_len = walker[0].log_density().len();
    debug_assert!(
        walker.iter()
            .all(|w| w.log_density().len() == log_len)
    );

    let mut averaged_log_density = walker[0].log_density_refined();
    norm_ln_prob(&mut averaged_log_density);

    if walker.len() > 1 {
    
        walker[1..].iter()
            .for_each(
                |w|
                {
                    let mut density = w.log_density_refined();
                    norm_ln_prob(&mut density);
                    averaged_log_density.iter_mut()
                        .zip(density.into_iter())
                        .for_each(
                            |(average, other)|
                            {
                                *average += other;
                            }
                        )
                }
            );
    
        let number_of_walkers = walker.len() as f64;
        averaged_log_density.iter_mut()
            .for_each(|average| *average /= number_of_walkers);
    }

    averaged_log_density
}