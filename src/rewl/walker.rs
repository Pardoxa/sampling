use rand::{Rng, SeedableRng, prelude::SliceRandom};
use std::{num::NonZeroUsize, marker::PhantomData, sync::*, mem::*};
use crate::*;
use crate::wang_landau::WangLandauMode;
use crate::glue_helper::*;

use rayon::prelude::*;

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

//five-point stencil 
fn five_point_derivitive(data: &[f64]) -> Vec<f64>
{
    let mut d = vec![f64::NAN; data.len()];
    if data.len() >= 5 {
        for i in 2..data.len()-2 {
            let mut tmp = data[i-1].mul_add(-8.0, data[i-2]);
            tmp = data[i+1].mul_add(8.0, tmp) - data[i+2];
            d[i] = tmp / 12.0;
        }
    }
    d
}

fn derivative(data: &[f64]) -> Vec<f64>
{
    let mut d = vec![f64::NAN; data.len()];
    if data.len() >= 3 {
        for i in 1..data.len()-1 {
            d[i] = (data[i+1] - data[i-1]) / 2.0;
        }
    }
    if data.len() >= 2 {
        d[0] = (data[1] - data[0]) / 2.0;

        d[data.len() - 1] = (data[data.len() - 1] - data[data.len() - 2]) / 2.0;
    }
    d
}

fn derivative_merged(data: &[f64]) -> Vec<f64>
{
    if data.len() < 5 {
        return derivative(data);
    }
    let mut d = five_point_derivitive(data);
    d[1] = (data[2] - data[0]) / 2.0;
    d[data.len() - 2] = (data[data.len() - 1] - data[data.len() - 3]) / 2.0;

    d[0] = (data[1] - data[0]) / 2.0;

    d[data.len() - 1] = (data[data.len() - 1] - data[data.len() - 2]) / 2.0;

    d
}


fn merge_walker_prob<R, Hist, Energy, S, Res>(walker: &mut [RewlWalker<R, Hist, Energy, S, Res>])
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

fn get_merged_walker_prob<R, Hist, Energy, S, Res>(walker: &[RewlWalker<R, Hist, Energy, S, Res>]) -> Vec<f64>
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


/// # Walker for Replica exchange Wang Landau
/// * used by [`Rewl`](`crate::rewl::Rewl`)
/// * performes the random walk in its respective domain 
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct RewlWalker<R, Hist, Energy, S, Res>
{
    id: usize,
    sweep_size: NonZeroUsize,
    rng: R,
    hist: Hist,
    log_density: Vec<f64>,
    log_f: f64,
    step_count: usize,
    mode: WangLandauMode,
    old_energy: Energy,
    bin: usize,
    marker_s: PhantomData<S>,
    marker_res: PhantomData<Res>,
}

fn replica_exchange<R, Hist, Energy, S, Res>
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
            marker_s: PhantomData::<S>
        }
    }

    /// # Current (logarithm of) factor f
    /// * See the paper for more info
    pub fn log_f(&self) -> f64
    {
        self.log_f
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
        }
    }
}


/// # Efficient replica exchange Wang landau
/// * use this to quickly build your own parallel replica exchange wang landau simulation
/// ## Tipp
/// Use the short hand `Rewl`  
#[derive(Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct ReplicaExchangeWangLandau<Ensemble, R, Hist, Energy, S, Res>{
    chunk_size: NonZeroUsize,
    ensembles: Vec<RwLock<Ensemble>>,
    walker: Vec<RewlWalker<R, Hist, Energy, S, Res>>,
    log_f_threshold: f64,
    replica_exchange_mode: bool,
}


/// Short for [`ReplicaExchangeWangLandau`](crate::rewl::ReplicaExchangeWangLandau)
pub type Rewl<Ensemble, R, Hist, Energy, S, Res> = ReplicaExchangeWangLandau<Ensemble, R, Hist, Energy, S, Res>;

impl<Ensemble, R, Hist, Energy, S, Res> Rewl<Ensemble, R, Hist, Energy, S, Res> 
where R: Send + Sync + Rng + SeedableRng,
    Hist: Send + Sync + Histogram + HistogramVal<Energy>,
    Energy: Send + Sync + Clone,
    Ensemble: MarkovChain<S, Res>,
    Res: Send + Sync,
    S: Send + Sync
{

    pub fn par_greed_from_ensemble<F>
    (
        ensemble: Ensemble,
        hists: Vec<Hist>,
        step_size: usize,
        sweep_size: NonZeroUsize,
        log_f_threshold: f64,
        chunk_size: NonZeroUsize,
        energy_fn: F
    ) -> Result<Self, RewlCreationErrors>
    where 
        Ensemble: Send + Sync + HasRng<R> + Clone,
        R: Send + Sync,
        F: Fn(&mut Ensemble) -> Option::<Energy> + Copy + Send + Sync,
        Hist: Clone
    {
        let size = NonZeroUsize::new(hists.len()).ok_or(RewlCreationErrors::EmptySlice)?;
        let ensembles = Self::clone_and_seed_ensembles(ensemble, size);
        Self::par_greed_from_ensemles_rng(ensembles, hists, step_size, sweep_size, log_f_threshold, chunk_size, energy_fn)
    }

    fn clone_and_seed_ensembles<R2>(mut ensemble: Ensemble, size: NonZeroUsize) -> Vec<Ensemble>
    where Ensemble: Clone + HasRng<R2>,
        R2: SeedableRng + Rng
    {
        let mut ensembles = Vec::with_capacity(size.get());
        ensembles.extend(
            (1..size.get())
                .map(|_| {
                    let mut e = ensemble.clone();
                    let mut rng = R2::from_rng(ensemble.rng())
                        .expect("unable to seed Rng");
                    e.swap_rng(&mut rng);
                    e
                })
        );
        ensembles.push(ensemble);
        ensembles
    }

    pub fn par_greed_from_ensembles<F>
    (
        ensemble: Vec<Ensemble>,
        hists: Vec<Hist>,
        step_size: usize,
        sweep_size: NonZeroUsize,
        log_f_threshold: f64,
        chunk_size: NonZeroUsize,
        energy_fn: F
    ) -> Result<Self, RewlCreationErrors>
    where 
        Ensemble: Send + Sync + HasRng<R> + Clone,
        R: Send + Sync,
        F: Fn(&mut Ensemble) -> Option::<Energy> + Copy + Send + Sync,
        Hist: Clone
    {
        Self::par_greed_from_ensemles_rng(ensemble, hists, step_size, sweep_size, log_f_threshold, chunk_size, energy_fn)
    }

    pub fn par_greed_from_ensemles_rng<F, R2>
    (
        ensembles: Vec<Ensemble>,
        hists: Vec<Hist>,
        step_size: usize,
        sweep_size: NonZeroUsize,
        log_f_threshold: f64,
        chunk_size: NonZeroUsize,
        energy_fn: F
    ) -> Result<Self, RewlCreationErrors>
    where 
        Ensemble: Send + Sync + HasRng<R2> + Clone,
        R: Send + Sync,
        F: Fn(&mut Ensemble) -> Option::<Energy> + Copy + Send + Sync,
        R2: Send + Sync + Rng + SeedableRng,
        Hist: Clone
    {
        

        if hists.len() != ensembles.len()
        {
            return Err(RewlCreationErrors::LenMissmatch);
        }

        let mut res = Vec::with_capacity(hists.len());

        ensembles.into_par_iter()
            .zip(hists.into_par_iter())
            .map(
                |(mut e, h)|
                {
                    let mut energy = loop{
                        if let Some(energy) = energy_fn(&mut e){
                            break energy;
                        }
                        e.m_steps_quiet(step_size);
                    };

                    if !h.is_inside(&energy) {
                        let mut distance = h.distance(&energy);

                        let mut steps = Vec::with_capacity(step_size);
                        loop {
                            e.m_steps(step_size, &mut steps);
                            let current_energy = if let Some(energy) = energy_fn(&mut e)
                            {
                                energy
                            } else {
                                e.undo_steps_quiet(&mut steps);
                                continue;
                            };

                            let new_distance = h.distance(&current_energy);
                            if new_distance <= distance {
                                energy = current_energy;
                                distance = new_distance;
                                if distance == 0.0 {
                                    break;
                                }
                            }else {
                                e.undo_steps_quiet(&mut steps);
                            }
                        }
                    }
                    (h, e, energy)
                }
            ).collect_into_vec(&mut res);

        let mut ensembles_rw_lock = Vec::with_capacity(res.len() * chunk_size.get());
        let mut walker = Vec::with_capacity(chunk_size.get() * res.len());

        let mut counter = 0;

        for (mut h, mut e, energy) in res.into_iter()
        {
            h.reset();
            for _ in 0..chunk_size.get()-1 {
                let mut ensemble = e.clone();
                let mut rng = R2::from_rng(e.rng())
                    .expect("unable to seed Rng");
                ensemble.swap_rng(&mut rng);
                
                ensembles_rw_lock.push(RwLock::new(ensemble));
                let rng = R::from_rng(e.rng())
                   .expect("unable to seed Rng");
                walker.push(
                    RewlWalker::new(
                        counter,
                        rng,
                        h.clone(),
                        sweep_size,
                        energy.clone()
                    )
                );
                counter += 1;
            }
            let rng = R::from_rng(e.rng())
                .expect("unable to seed Rng");
            walker.push(
                RewlWalker::new(
                    counter,
                    rng,
                    h,
                    sweep_size,
                    energy
                )
            );
            counter += 1;
            ensembles_rw_lock.push(RwLock::new(e));

        }

        Ok(
            Self
            {
                ensembles: ensembles_rw_lock,
                replica_exchange_mode: true,
                chunk_size,
                walker,
                log_f_threshold
            }
        )


    }

    pub fn simulate_until_convergence<F>(
        &mut self,
        step_size: usize,
        energy_fn: F
    )
    where 
        Ensemble: Send + Sync,
        R: Send + Sync,
        F: Fn(&mut Ensemble) -> Option<Energy> + Copy + Send + Sync
    {
        while !self.is_finished()
        {
            self.sweep(step_size, energy_fn);
        }
    }

    pub fn sweep<F>(&mut self, step_size: usize, energy_fn: F)
    where Ensemble: Send + Sync,
        R: Send + Sync,
        F: Fn(&mut Ensemble) -> Option<Energy> + Copy + Send + Sync
    {
        let slice = self.ensembles.as_slice();
        self.walker
            .par_iter_mut()
            .for_each(|w| w.wang_landau_sweep(slice, step_size, energy_fn));

        
        self.walker
            .par_chunks_mut(self.chunk_size.get())
            .filter(|chunk| 
                {
                    chunk.iter()
                        .all(RewlWalker::all_bins_reached)
                }
            )
            .for_each(
                |chunk|
                {
                    chunk.iter_mut()
                        .for_each(RewlWalker::refine_f_reset_hist);
                    merge_walker_prob(chunk);
                }
            );

        // replica exchange
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

    pub fn is_finished(&self) -> bool
    {
        self.walker
            .iter()
            .all(|w| w.log_f < self.log_f_threshold)
    }

    /// # Result of the simulations!
    /// This is what we do the simulation for!
    /// 
    /// It returns the log10 of the normalized probability density and the 
    /// histogram, which contains the corresponding bins.
    ///
    /// Failes if the internal histograms (invervals) do not align. Might fail if 
    /// there is no overlap between neighboring intervals 
    pub fn merged_log10_prob(&self) -> Result<(Vec<f64>, Hist), HistErrors>
    where Hist: HistogramCombine
    {
        let (mut log_prob, e_hist) = self.merged_log_prob()?;

        // switch base of log
        log_prob.iter_mut()
            .for_each(|val| *val *= std::f64::consts::LOG10_E);

        Ok((log_prob, e_hist))

    }

    pub fn merged_log_prob(&self) -> Result<(Vec<f64>, Hist), HistErrors>
    where Hist: HistogramCombine
    {
        let (mut log_prob, e_hist) = self.merged_log_probability_helper()?;

        subtract_max(&mut log_prob);

        // calculate actual sum in non log space
        let sum = log_prob.iter()
            .fold(0.0, |acc, &val| {
                if val.is_finite(){
                   acc +  val.exp()
                } else {
                    acc
                }
            }  
        );

        let sum = sum.ln();

        log_prob.iter_mut()
            .for_each(|val| *val -= sum);
        
        Ok((log_prob, e_hist))
    }

    fn merged_log_probability_helper(&self) -> Result<(Vec<f64>, Hist), HistErrors>
    where Hist: HistogramCombine
    {
        // get the log_probabilities - the walkers over the same intervals are merged
        let mut log_prob: Vec<_> = self.walker
            .par_chunks(self.chunk_size.get())
            .map(get_merged_walker_prob)
            .collect();
        
        log_prob
            .par_iter_mut()
            .for_each(|v| subtract_max(v));


        // get the derivative, for merging later
        let derivatives: Vec<_> = log_prob.par_iter()
            .map(|v| derivative_merged(v))
            .collect();

        let hists: Vec<_> = self.walker.iter()
            .step_by(self.chunk_size.get())
            .map(|w| &w.hist)
            .collect();

        let e_hist = Hist::encapsulating_hist(&hists)?;

        let alignment  = hists.iter()
            .zip(hists.iter().skip(1))
            .map(|(&left, &right)| left.align(right))
            .collect::<Result<Vec<_>, _>>()?;
        
        
        let merge_points: Vec<_> = derivatives.iter()
            .zip(derivatives.iter().skip(1))
            .zip(alignment.iter())
            .map(
                |((left, right), &align)|
                {
                    (align..)
                        .zip(
                            left[align..].iter()
                            .zip(right.iter())
                        )
                        .map(
                            |(index, (&left, &right))|
                            {
                                (index, (left - right).abs())
                            }
                        ).fold( (usize::MAX, f64::INFINITY),
                            |a, b|
                            if a.1 < b.1 {
                                a
                            } else {
                                b
                            }
                        ).0
                }
            ).collect();

        let mut merged_log_prob = vec![f64::NAN; e_hist.bin_count()];
        
        merged_log_prob[..=merge_points[0]]
            .copy_from_slice(&log_prob[0][..=merge_points[0]]);


        // https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=2dcb7b7a3be78397d34657ece42aa851
        let mut align_sum = 0;
        for (index, (&a, &mp)) in alignment.iter().zip(merge_points.iter()).enumerate()
        {
            let position_l = mp + align_sum;
            align_sum += a;
            let left = mp - a;

            let shift = merged_log_prob[position_l] - log_prob[index + 1][left];

            merged_log_prob[position_l..]
                .iter_mut()
                .zip(log_prob[index + 1][left..].iter())
                .for_each(
                    |(merge, val)|
                    {
                        *merge = val + shift;
                    }
                );


        }

        Ok((merged_log_prob, e_hist))

    }

    /// # Get Ids
    /// This is an indicator that the replica exchange works.
    /// In the beginning, this will be a sorted vector, e.g. [0,1,2,3,4].
    /// Then it will show, where the ensemble, which the corresponding walkers currently work with,
    /// originated from. E.g. If the vector is [3,1,0,2,4], Then walker 0 has a
    /// ensemble originating from walker 3, the walker 1 is back to its original 
    /// ensemble, walker 2 has an ensemble originating form walker 0 and so on.
    pub fn get_id_vec(&self) -> Vec<usize>
    {
        self.walker
            .iter()
            .map(|w| w.id)
            .collect()
    }
}
