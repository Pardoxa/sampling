use rand::Rng;
use std::{marker::PhantomData, mem::*, num::*, sync::*, usize};
use crate::{*, rewl::log_density_to_log10_density};

#[cfg(feature = "sweep_time_optimization")]
use std::time::*;

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

/// # Walker for Replica exchange entropic sampling
/// * performes the random walk in its respective domain 
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct ReesWalker<R, Hist, Energy, S, Res>
{
    id: usize,
    sweep_size: NonZeroUsize,
    pub(crate) rng: R,
    hist: Hist,
    log_density: Vec<f64>,
    step_count: usize,
    re: usize,
    proposed_re: usize,
    old_energy: Energy,
    bin: usize,
    markov_steps: Vec<S>,
    marker_res: PhantomData<Res>,
    step_size: usize,
    step_threshold: usize,
    #[cfg(feature = "sweep_time_optimization")]
    duration: Duration
}

impl<R, Hist, Energy, S, Res> From<RewlWalker<R, Hist, Energy, S, Res>> for ReesWalker<R, Hist, Energy, S, Res>
where Hist: Histogram
{
    fn from(mut rewl_walker: RewlWalker<R, Hist, Energy, S, Res>) -> Self
    {
        rewl_walker.hist.reset();
        Self{
            id: rewl_walker.id,
            sweep_size: rewl_walker.sweep_size,
            markov_steps: rewl_walker.markov_steps,
            step_size: rewl_walker.step_size,
            marker_res: PhantomData::<Res>,
            rng: rewl_walker.rng,
            log_density: rewl_walker.log_density,
            bin: rewl_walker.bin,
            hist: rewl_walker.hist,
            old_energy: rewl_walker.old_energy,
            step_count: 0,
            step_threshold: rewl_walker.step_count,
            re: 0,
            proposed_re: 0,
            #[cfg(feature = "sweep_time_optimization")]
            duration: rewl_walker.duration,
        }    
    }
}

impl<R, Hist, Energy, S, Res> ReesWalker<R, Hist, Energy, S, Res>
{
    /// # Returns id of walker
    /// * important for mapping the ensemble to the walker
    pub fn id(&self) -> usize
    {
        self.id
    }

    /// # Returns duration of last sweep that was performed
    #[cfg(feature = "sweep_time_optimization")]
    pub fn duration(&self) -> Duration
    {
        self.duration
    }

    /// # Reference to internal histogram
    pub fn hist(&self) -> &Hist
    {
        &self.hist
    }

    /// # how many steps per sweep
    pub fn sweep_size(&self) -> NonZeroUsize
    {
        self.sweep_size
    }

    /// # change how many steps per sweep are performed
    pub fn sweep_size_change(&mut self, sweep_size: NonZeroUsize)
    {
        self.sweep_size = sweep_size;
    }

    /// # step size for markov steps
    pub fn step_size(&self) -> usize 
    {
        self.step_size
    }

    /// # Change step sitze for markov steps
    pub fn step_size_change(&mut self, step_size: usize)
    {
        self.step_size = step_size;
    }

    /// # How many entropic steps were performed until now?
    pub fn step_count(&self) -> usize
    {
        self.step_count
    }

    /// # How many successful replica exchanges were performed until now?
    pub fn replica_exchanges(&self) -> usize
    {
        self.re
    }

    /// # How many replica exchanges were proposed until now?
    pub fn proposed_replica_exchanges(&self) -> usize
    {
        self.proposed_re
    }

    /// fraction of how many replica exchanges were accepted and how many were proposed
    pub fn replica_exchange_frac(&self) -> f64
    {
        self.re as f64 / self.proposed_re as f64
    }

    /// * Old non normalized estimate of the natural logarithm of the probability density function
    /// * for refined density use `self.log_density_refined()`
    pub fn log_density(&self) -> &[f64]
    {
        &self.log_density
    }

    /// * Current non normalized estimate of the natural logarithm of the probability density function
    /// * calculated by refining old density with current histogram
    pub fn log_density_refined(&self) -> Vec<f64>
    where Hist: Histogram
    {
        let mut refined_log_density = Vec::with_capacity(self.log_density.len());

        refined_log_density.extend(
            self.log_density
                .iter()
                .zip(self.hist.hist().iter())
                .map(
                    |(&log_d, &h)|
                    {
                        if h == 0 {
                            log_d
                        } else {
                            log_d + (h as f64).ln()
                        }
                    }
                )
        ); 
        refined_log_density
    }

    /// # Old estimate of log10 of probability density
    /// * normalized (sum over non log values is 1 (within numerical precision))
    pub fn log10_density(&self) -> Vec<f64>
    {
        log_density_to_log10_density(self.log_density())
    }

    /// # Current estimate of log10 of probability density
    /// * normalized (sum over non log values is 1 (within numerical precision))
    pub fn log10_density_refined(&self) -> Vec<f64>
    where Hist: Histogram
    {
        let density = self.log_density_refined();
        log_density_to_log10_density(&density)

    }

    /// # is the simulation finished?
    /// * true, if more steps than the step threshold are performed
    #[inline(always)]
    pub fn is_finished(&self) -> bool
    {
        self.step_count >= self.step_threshold
    }

    pub fn refine(&mut self)
    where Hist: Histogram
    {
        let refined = self.log_density_refined();
        self.log_density = refined;
        self.hist.reset();
        self.step_count = 0;
    }
}

impl<R, Hist, Energy, S, Res> ReesWalker<R, Hist, Energy, S, Res>
where Hist: HistogramVal<Energy>,
    R: Rng
{
    pub(crate) fn sweep<Ensemble, F, Extra, P>
    (
        &mut self,
        ensemble_vec: &[RwLock<Ensemble>],
        extra: &mut Extra,
        extra_fn: P,
        energy_fn: F,
    )
    where F: Fn(&mut Ensemble) -> Option<Energy>,
        P: Fn(&Self, &mut Ensemble, &mut Extra) -> (),
        Ensemble: MarkovChain<S, Res>,
    {
        #[cfg(feature = "sweep_time_optimization")]
        let start = Instant::now();

        let mut e = ensemble_vec[self.id]
            .write()
            .expect("Fatal Error encountered; ERRORCODE 0x1 - this should be \
                impossible to reach. If you are using the latest version of the \
                'sampling' library, please contact the library author via github by opening an \
                issue! https://github.com/Pardoxa/sampling/issues");
        
        for _ in 0..self.sweep_size.get()
        {   
            self.step_count = self.step_count.saturating_add(1);
            e.m_steps(self.step_size, &mut self.markov_steps);

            let energy = match energy_fn(&mut e){
                Some(energy) => energy,
                None => {
                    e.undo_steps_quiet(&self.markov_steps);
                    continue;
                }
            };


            match self.hist.get_bin_index(&energy) 
            {
                Ok(current_bin) => {
                    // metropolis hastings
                    let acception_prob = (self.log_density[self.bin] - self.log_density[current_bin])
                        .exp();
                    if self.rng.gen::<f64>() > acception_prob 
                    {
                        e.undo_steps_quiet(&self.markov_steps);
                    } else {
                        self.old_energy = energy;
                        self.bin = current_bin;
                    }
                },
                _ => {
                    e.undo_steps_quiet(&self.markov_steps);
                }
            }

            self.hist.count_index(self.bin)
                .expect("Histogram index Error, ERRORCODE 0x2");

            extra_fn(&self, &mut e, extra);

        }
        #[cfg(feature = "sweep_time_optimization")]
            {
                self.duration = start.elapsed();
            }
    }
}

pub(crate) fn replica_exchange<R, Hist, Energy, S, Res>
(
    walker_a: &mut ReesWalker<R, Hist, Energy, S, Res>,
    walker_b: &mut ReesWalker<R, Hist, Energy, S, Res>
) where Hist: HistogramVal<Energy>,
    R: Rng
{
    walker_a.proposed_re += 1;
    walker_b.proposed_re += 1;
    // check if exchange is even possible
    let new_bin_a = match walker_a.hist.get_bin_index(&walker_b.old_energy)
    {
        Ok(bin) => bin,
        _ => return,
    };

    let new_bin_b = match walker_b.hist.get_bin_index(&walker_a.old_energy)
    {
        Ok(bin) => bin,
        _ => return,
    };

    // see paper equation 1
    let log_gi_x = walker_a.log_density[walker_a.bin];
    let log_gi_y = walker_a.log_density[new_bin_a];

    let log_gj_y = walker_b.log_density[walker_b.bin];
    let log_gj_x = walker_b.log_density[new_bin_b];

    let log_prob = log_gi_x + log_gj_y - log_gi_y - log_gj_x;

    let prob = log_prob.exp();

    // if exchange is accepted
    if walker_b.rng.gen::<f64>() < prob 
    {
        swap(&mut walker_b.id, &mut walker_a.id);
        swap(&mut walker_b.old_energy, &mut walker_a.old_energy);
        walker_b.bin = new_bin_b;
        walker_a.bin = new_bin_a;
        walker_a.re += 1;
        walker_b.re += 1;
    }
}

pub(crate) fn get_merged_refined_walker_prob<R, Hist, Energy, S, Res>(walker: &[ReesWalker<R, Hist, Energy, S, Res>]) -> Vec<f64>
where Hist: Histogram
{
    let log_len = walker[0].log_density.len();
    debug_assert!(
        walker.iter()
            .all(|w| w.log_density.len() == log_len)
    );

    let mut averaged_log_density = walker[0].log_density_refined();

    if walker.len() > 1 {

        walker[1..]
            .iter()
            .map(|w| w.log_density_refined())
            .for_each(
                |log_density|
                {
                    averaged_log_density.iter_mut()
                        .zip(log_density.into_iter())
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