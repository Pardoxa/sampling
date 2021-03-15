use rand::Rng;
use std::{marker::PhantomData, mem::*, num::*, sync::*, usize};
use crate::*;
use crate::wang_landau::WangLandauMode;

#[cfg(feature = "sweep_time_optimization")]
use std::time::*;

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
/// Errors encountered during the creation of a Rewl struct (**R**eplica **e**xchange **W**ang **L**andau)
pub enum RewlCreationErrors
{
    /// histograms must have at least two bins - everything else makes no sense!
    HistsizeError,

    /// You tried to pass an empty slice
    EmptySlice,

    /// The length of the histogram vector has to be equal to the length of the ensemble vector!
    LenMissmatch,
}


pub(crate) fn log_density_to_log10_density(log_density: &[f64]) -> Vec<f64>
{

    let max = log_density.iter()
        .fold(f64::NEG_INFINITY,  |acc, &val| acc.max(val));
    let mut log_density_res: Vec<f64> = Vec::with_capacity(log_density.len());
    log_density_res.extend(
        log_density.iter()
            .map(|&val| val - max)
    );
    
    let sum = log_density_res.iter()
        .fold(0.0, |acc, &val| 
            {
                if val.is_finite(){
                    acc +  val.exp()
                } else {
                    acc
                }
            }
        );
    let sum = -sum.log10();

    log_density_res.iter_mut()
        .for_each(|val| *val = val.mul_add(std::f64::consts::LOG10_E, sum));
    log_density_res
            
    
}


/// # Walker for Replica exchange Wang Landau
/// * used by [`Rewl`](`crate::rewl::Rewl`)
/// * performes the random walk in its respective domain 
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct RewlWalker<R, Hist, Energy, S, Res>
{
    pub(crate) id: usize,
    pub(crate) sweep_size: NonZeroUsize,
    pub(crate) rng: R,
    pub(crate) hist: Hist,
    pub(crate) log_density: Vec<f64>,
    log_f: f64,
    pub(crate) step_count: usize,
    pub(crate) proposed_re: usize,
    pub(crate) re: usize,
    mode: WangLandauMode,
    pub(crate) old_energy: Energy,
    pub(crate) bin: usize,
    pub(crate) markov_steps: Vec<S>,
    marker_res: PhantomData<Res>,
    pub(crate) step_size: usize,
    #[cfg(feature = "sweep_time_optimization")]
    pub(crate) duration: Duration,
    #[cfg(feature = "sweep_stats")]
    pub(crate) sweep_stats: SweepStats,
}

impl<R, Hist, Energy, S, Res> RewlWalker<R, Hist, Energy, S, Res>{
    /// # Returns id of walker
    /// * important for mapping the ensemble to the walker
    pub fn id(&self) -> usize
    {
        self.id
    }

    pub fn wang_landau_mode(&self) -> WangLandauMode
    {
        self.mode
    }

    /// # Returns duration of last sweep that was performed
    #[cfg(feature = "sweep_time_optimization")]
    pub fn duration(&self) -> Duration
    {
        self.duration
    }

    #[cfg(feature = "sweep_stats")]
    pub fn average_sweep_duration(&self) -> Duration
    {
        self.sweep_stats.averag_duration()
    }

    #[cfg(feature = "sweep_stats")]
    pub fn high_low_10_percent(&self) -> (Duration, Duration)
    {
        self.sweep_stats.percent_high_low()
    }

    #[cfg(feature = "sweep_stats")]
    pub fn last_durations(&self) -> &[Duration]
    {
        self.sweep_stats.buf()
    }

    /// Returns reference of current energy
    pub fn energy(&self) -> &Energy
    {
        &self.old_energy
    }

    /// Returns current energy
    pub fn energy_copy(&self) -> Energy
    where Energy: Copy
    {
        self.old_energy
    }

    /// Returns current energy
    pub fn energy_clone(&self) -> Energy
    where Energy: Clone
    {
        self.old_energy.clone()
    }

    /// # Reference to internal histogram
    pub fn hist(&self) -> &Hist
    {
        &self.hist
    }

    /// # Current (logarithm of) factor f
    /// * See the paper for more info
    pub fn log_f(&self) -> f64
    {
        self.log_f
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

    /// # How many steps were performed until now?
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

    /// Current non normalized estimate of the natural logarithm of the probability density function
    pub fn log_density(&self) -> &[f64]
    {
        &self.log_density
    }
}


impl<R, Hist, Energy, S, Res> RewlWalker<R, Hist, Energy, S, Res> 
where R: Rng + Send + Sync,
    Self: Send + Sync,
    Hist: Histogram + HistogramVal<Energy>,
    Energy: Send + Sync,
    S: Send + Sync,
    Res: Send + Sync
{
    pub(crate) fn new
    (
        id: usize,
        rng: R,
        hist: Hist,
        sweep_size: NonZeroUsize,
        step_size: usize,
        old_energy: Energy,
    ) -> RewlWalker<R, Hist, Energy, S, Res>
    {
        let log_density = vec![0.0; hist.bin_count()];
        let bin = hist.get_bin_index(&old_energy).unwrap();
        let markov_steps = Vec::with_capacity(step_size);
        RewlWalker{
            id,
            rng,
            hist,
            log_density,
            sweep_size,
            log_f: 1.0,
            step_count: 0,
            re: 0,
            proposed_re: 0,
            mode: WangLandauMode::RefineOriginal,
            old_energy,
            bin,
            marker_res: PhantomData::<Res>,
            markov_steps,
            step_size,
            #[cfg(feature = "sweep_time_optimization")]
            duration: Duration::from_millis(0),
            #[cfg(feature = "sweep_stats")]
            sweep_stats: SweepStats::new(),
        }
    }

    

    /// # Current estimate of log10 of probability density
    /// * normalized (sum over non log values is 1 (within numerical precision))
    pub fn log10_density(&self) -> Vec<f64>
    {
        log_density_to_log10_density(self.log_density())
    }

    fn log_f_1_t(&self) -> f64
    {
        self.hist.bin_count() as f64 / self.step_count as f64
    }

    pub(crate) fn all_bins_reached(&self) -> bool
    {
        !self.hist.any_bin_zero()
    }

    pub(crate) fn refine_f_reset_hist(&mut self)
    {
        // Check if log_f should be halfed or mode should be changed
        if self.mode.is_mode_original() && !self.hist.any_bin_zero() {
            let ref_1_t = self.log_f_1_t();
            self.log_f *= 0.5;

            if self.log_f < ref_1_t {
                self.log_f = ref_1_t;
                self.mode = WangLandauMode::Refine1T;
            }
        }
        self.hist.reset();
    }

    pub(crate) fn check_energy_fn<F, Ensemble>(
        &self,
        ensemble_vec: &[RwLock<Ensemble>],
        energy_fn: F
    )   -> bool
    where Energy: PartialEq,F: Fn(&mut Ensemble) -> Option<Energy>,
        
    {
        let mut e = ensemble_vec[self.id]
            .write()
            .expect("Fatal Error encountered; ERRORCODE 0x5 - this should be \
                impossible to reach. If you are using the latest version of the \
                'sampling' library, please contact the library author via github by opening an \
                issue! https://github.com/Pardoxa/sampling/issues");
        
        let energy = match energy_fn(&mut e){
            Some(energy) => energy,
            None => {
                return false;
            }
        };
        energy == self.old_energy
    }

    pub(crate) fn wang_landau_sweep<Ensemble, F>
    (
        &mut self,
        ensemble_vec: &[RwLock<Ensemble>],
        energy_fn: F
    )
    where F: Fn(&mut Ensemble) -> Option<Energy>,
        Ensemble: MarkovChain<S, Res>
    {
        #[cfg(feature = "sweep_time_optimization")]
        let start = Instant::now();

        let mut e = ensemble_vec[self.id]
            .write()
            .expect("Fatal Error encountered; ERRORCODE 0x6 - this should be \
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

            
            if self.mode.is_mode_1_t() {
                self.log_f = self.log_f_1_t();
            }

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
                .expect("Histogram index Error, ERRORCODE 0x7");
            
            self.log_density[self.bin] += self.log_f;

        }
        #[cfg(feature = "sweep_time_optimization")]
            {
                self.duration = start.elapsed();
                #[cfg(feature = "sweep_stats")]
                self.sweep_stats.push(self.duration);
            }
    }
}


pub(crate) fn merge_walker_prob<R, Hist, Energy, S, Res>(walker: &mut [RewlWalker<R, Hist, Energy, S, Res>])
{
    
    if walker.len() < 2 {
        return;
    }
    let averaged = get_merged_walker_prob(walker);
    
    walker.iter_mut()
        .skip(1)
        .for_each(
            |w|
            {
                w.log_density
                    .copy_from_slice(&averaged)
            }
        );
    walker[0].log_density = averaged;
}

pub(crate) fn get_merged_walker_prob<R, Hist, Energy, S, Res>(walker: &[RewlWalker<R, Hist, Energy, S, Res>]) -> Vec<f64>
{
    let log_len = walker[0].log_density.len();
    debug_assert!(
        walker.iter()
            .all(|w| w.log_density.len() == log_len)
    );

    let mut averaged_log_density = walker[0].log_density
        .clone();

    if walker.len() > 1 {
    
        walker[1..].iter()
            .for_each(
                |w|
                {
                    averaged_log_density.iter_mut()
                        .zip(w.log_density.iter())
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

pub(crate) fn replica_exchange<R, Hist, Energy, S, Res>
(
    walker_a: &mut RewlWalker<R, Hist, Energy, S, Res>,
    walker_b: &mut RewlWalker<R, Hist, Energy, S, Res>
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
        walker_b.re +=1;
        walker_a.re +=1;
    }
}