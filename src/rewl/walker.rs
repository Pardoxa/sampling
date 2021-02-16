use rand::Rng;
use std::{marker::PhantomData, mem::*, num::NonZeroUsize, sync::*, usize};
use crate::*;
use crate::wang_landau::WangLandauMode;

#[cfg(feature = "rewl_sweep_time")]
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



/// # Walker for Replica exchange Wang Landau
/// * used by [`Rewl`](`crate::rewl::Rewl`)
/// * performes the random walk in its respective domain 
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct RewlWalker<R, Hist, Energy, S, Res>
{
    id: usize,
    sweep_size: NonZeroUsize,
    pub(crate) rng: R,
    hist: Hist,
    log_density: Vec<f64>,
    log_f: f64,
    step_count: usize,
    mode: WangLandauMode,
    old_energy: Energy,
    bin: usize,
    marker_s: PhantomData<S>,
    marker_res: PhantomData<Res>,
    #[cfg(feature = "rewl_sweep_time")]
    duration: Duration
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
        old_energy: Energy
    ) -> RewlWalker<R, Hist, Energy, S, Res>
    {
        let log_density = vec![0.0; hist.bin_count()];
        let bin = hist.get_bin_index(&old_energy).unwrap();
        RewlWalker{
            id,
            rng,
            hist,
            log_density,
            sweep_size,
            log_f: 1.0,
            step_count: 0,
            mode: WangLandauMode::RefineOriginal,
            old_energy,
            bin,
            marker_res: PhantomData::<Res>,
            marker_s: PhantomData::<S>,
            #[cfg(feature = "rewl_sweep_time")]
            duration: Duration::from_millis(0)
        }
    }

    /// # Returns id of walker
    /// * important for mapping the ensemble to the walker
    pub fn id(&self) -> usize
    {
        self.id
    }

    /// # Returns duration of last sweep that was performed
    #[cfg(feature = "rewl_sweep_time")]
    pub fn duration(&self) -> Duration
    {
        self.duration
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

    /// Current non normalized estimate of the natural logarithm of the probability density function
    pub fn log_density(&self) -> &[f64]
    {
        &self.log_density
    }

    /// # Current estimate of log10 of probability density
    /// * normalized (sum over non log values is 1 (within numerical precision))
    pub fn log10_density(&self) -> Vec<f64>
    {

        let max = self.log_density.iter()
            .fold(f64::NEG_INFINITY,  |acc, &val| acc.max(val));
        let mut log_density: Vec<f64> = Vec::with_capacity(self.log_density.len());
        log_density.extend(
            self.log_density.iter()
                .map(|&val| val - max)
        );
        
        let sum = log_density.iter()
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

        log_density.iter_mut()
            .for_each(|val| *val = val.mul_add(std::f64::consts::LOG10_E, sum));

        log_density
            
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

    pub(crate) fn wang_landau_sweep<Ensemble, F>
    (
        &mut self,
        ensemble_vec: &[RwLock<Ensemble>],
        step_size: usize,
        energy_fn: F
    )
    where F: Fn(&mut Ensemble) -> Option<Energy>,
        Ensemble: MarkovChain<S, Res>
    {
        #[cfg(feature = "rewl_sweep_time")]
        let start = Instant::now();

        let mut e = ensemble_vec[self.id]
            .write()
            .expect("Fatal Error encountered; ERRORCODE 0x1 - this should be \
                impossible to reach. If you are using the latest version of the \
                'sampling' library, please contact the library author via github by opening an \
                issue! https://github.com/Pardoxa/sampling/issues");
        
        let mut steps = Vec::with_capacity(step_size);
        for _ in 0..self.sweep_size.get()
        {   
            self.step_count = self.step_count.saturating_add(1);
            e.m_steps(step_size, &mut steps);

            let energy = match energy_fn(&mut e){
                Some(energy) => energy,
                None => {
                    e.undo_steps_quiet(&mut steps);
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
                        e.undo_steps_quiet(&steps);
                    } else {
                        self.old_energy = energy;
                        self.bin = current_bin;
                    }
                },
                _ => {
                    e.undo_steps_quiet(&steps);
                }
            }

            self.hist.count_index(self.bin)
                .expect("Histogram index Error, ERRORCODE 0x2");
            
            self.log_density[self.bin] += self.log_f;

            #[cfg(feature = "rewl_sweep_time")]
            {
                self.duration = start.elapsed();
            }

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
    }
}