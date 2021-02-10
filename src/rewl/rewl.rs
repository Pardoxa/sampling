use crate::*;
use crate::rewl::*;
use crate::glue_helper::*;
use rand::{Rng, SeedableRng, prelude::SliceRandom, Error};
use std::{marker::PhantomData, num::NonZeroUsize, sync::*};
use rayon::prelude::*;

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};


/// # Use this to create a replica exchange wang landau simulation
/// * Tipp: Use shorthand `RewlBuilder`
#[derive(Debug,Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct ReplicaExchangeWangLandauBuilder<Ensemble, Hist, S, Res>
{
    chunk_size: NonZeroUsize,
    ensembles: Vec<Ensemble>,
    hists: Vec<Hist>,
    log_f_threshold: f64,
    sweep_size: NonZeroUsize,
    step_size: NonZeroUsize,
    res: PhantomData<Res>,
    s: PhantomData<S>
}

/// # Short for `ReplicaExchangeWangLandauBuilder`
pub type RewlBuilder<Ensemble, Hist, S, Res> = ReplicaExchangeWangLandauBuilder<Ensemble, Hist, S, Res>;

#[derive(Debug)]
pub enum RewlBuilderErr{
    /// * The threshold for `log_f` needs to be a normal number.
    /// * That basically means: the number is neither zero, infinite, subnormal, or NaN. 
    /// For more info, see the [Documentation](`std::primitive::f64::is_normal`)
    NonNormalThreshold,
    /// Histogram vector needs to contain at least one entry.
    Empty,
    /// Each histogram needs to have **at least** two bins. Though more than two bins are 
    /// strongly recommended
    HistBinCount,

    /// Unable to seed random number generator 
    SeedError(Error),

    /// Length of histogram vector and ensemble vector has to be the same!
    LenMissmatch
}

impl<Ensemble, Hist, S, Res> RewlBuilder<Ensemble, Hist, S, Res>
where Hist: Histogram,
    Ensemble: MarkovChain<S, Res> + Sized + Sync + Send + Clone,
    Hist: Sized + Sync + Send + Clone,
    S: Send + Sync,
    Res: Sync + Send
{
    pub fn new(
        ensembles: Vec<Ensemble>,
        hists: Vec<Hist>,
        step_size: NonZeroUsize,
        sweep_size: NonZeroUsize,
        chunk_size: NonZeroUsize,
        log_f_threshold: f64
    ) -> Result<Self, RewlBuilderErr>
    {
        if !log_f_threshold.is_normal(){
            return Err(RewlBuilderErr::NonNormalThreshold);
        }
        let log_f_threshold = log_f_threshold.abs();
        if hists.len() == 0 
        {
            return Err(RewlBuilderErr::Empty);
        }
        if hists.len() != ensembles.len()
        {
            return Err(RewlBuilderErr::LenMissmatch);
        }
        if hists.iter().any(|v| v.bin_count() < 2)
        {
            return Err(RewlBuilderErr::HistBinCount);
        }

        Ok(
            Self{
                ensembles,
                step_size,
                sweep_size,
                chunk_size,
                hists,
                log_f_threshold,
                s: PhantomData::<S>,
                res: PhantomData::<Res>
            }
        )
    }
    
    /// # Create a builder to create a replica exchange wang landau (Rewl) simulation
    /// * creates vector of ensembles and (re)seeds their respective rngs (by using the `HasRng` trait)
    /// * calls [`Self::new(…)`](`crate::ReplicaExchangeWangLandauBuilder::new`) afterwards
    pub fn new_clone_ensemble<R>(
        ensemble: Ensemble,
        hists: Vec<Hist>,
        step_size: NonZeroUsize,
        sweep_size: NonZeroUsize,
        chunk_size: NonZeroUsize,
        log_f_threshold: f64,

    ) -> Result<Self,RewlBuilderErr>
    where Ensemble: HasRng<R> + Clone,
        R: Rng + SeedableRng
    {
        let len = NonZeroUsize::new(hists.len())
            .ok_or(RewlBuilderErr::Empty)?;
        let ensembles = Self::clone_and_seed_ensembles(ensemble, len)?;
        Self::new(ensembles, hists, step_size, sweep_size, chunk_size, log_f_threshold)
    }

    fn clone_and_seed_ensembles<R>(mut ensemble: Ensemble, size: NonZeroUsize) -> Result<Vec<Ensemble>, RewlBuilderErr>
    where Ensemble: Clone + HasRng<R>,
        R: SeedableRng + Rng
    {
         let mut ensembles = (1..size.get())
                .map(|_| {
                    let mut e = ensemble.clone();
                    let mut rng = R::from_rng(ensemble.rng())?;
                    e.swap_rng(&mut rng);
                    Ok(e)
                })
                .collect::<Result<Vec<_>,Error>>()
                .map_err(|e| RewlBuilderErr::SeedError(e))?;
        ensembles.push(ensemble);
        Ok(ensembles)
    }

    //fn greedy_sweep<F, Energy>(
    //    ensemble: &mut Ensemble,
    //    hist: &Hist,
    //    sweep_size: NonZeroUsize,
    //    step_size: NonZeroUsize,
    //    steps: &mut Vec<S>,
    //    energy_fn: F,
    //    distance: &mut f64,
    //    energy: &mut Energy
    //)where F: Fn(&mut Ensemble) -> Option::<Energy> + Copy + Send + Sync,
    //Hist: HistogramVal<Energy>,
    //{
    //    for _ in 0..sweep_size.get()
    //    {
    //        ensemble.m_steps(step_size.get(), &mut steps);
    //        let current_energy = if let Some(energy) = energy_fn(&mut ensemble)
    //        {
    //            energy
    //        } else {
    //            ensemble.undo_steps_quiet(&mut steps);
    //            continue;
    //        };
//
    //        let new_distance = hist.distance(&current_energy);
    //        if new_distance <= *distance {
    //            energy = current_energy;
    //            *distance = new_distance;
    //            if *distance == 0.0 {
    //                break 'outer2;
    //            }
    //        }else {
    //            e.undo_steps_quiet(&mut steps);
    //        }
    //    }
    //}

    fn build<Energy, R, R2>
    (
        container: Vec<(Hist, Ensemble, Option<Energy>)>,
        chunk_size: NonZeroUsize,
        log_f_threshold: f64,
        step_size: NonZeroUsize,
        sweep_size: NonZeroUsize

    ) -> Result<Rewl<Ensemble, R, Hist, Energy, S, Res>, Self>
    where Energy: Clone,
    R2: Rng + SeedableRng,
    Ensemble: HasRng<R2>,
    R: SeedableRng + Rng + Send + Sync,
    walker::RewlWalker<R, Hist, Energy, S, Res>: Send +  Sync,
    Hist: HistogramVal<Energy>,
    Energy: Send + Sync
    {
        if container.iter().any(|(_, _, e)| e.is_none()){
            let (hists, ensembles) = container.into_iter()
                .map(|(h, e, _)| (h, e))
                .unzip();
            return Err(
                Self{
                    ensembles,
                    hists,
                    chunk_size,
                    s: PhantomData::<S>,
                    res: PhantomData::<Res>,
                    log_f_threshold,
                    step_size,
                    sweep_size
                }
            );
        }

        let mut ensembles_rw_lock = Vec::with_capacity(container.len() * chunk_size.get());
        let mut walker = Vec::with_capacity(chunk_size.get() * container.len());
        let mut counter = 0;

        for (mut h, mut e, energy) in container.into_iter()
        {
            let energy = energy.unwrap();
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
                    RewlWalker::<R, Hist, Energy, S, Res>::new(
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
            Rewl{
                ensembles: ensembles_rw_lock,
                replica_exchange_mode: true,
                chunk_size,
                walker,
                log_f_threshold
            }
        )
    }
    
    pub fn try_greedy_build<R, F, C, Energy>(self, energy_fn: F, condition: C) -> Result<Rewl<Ensemble, R, Hist, Energy, S, Res>, Self>
    where Hist: HistogramVal<Energy>,
        Ensemble: HasRng<R> + Sized,
        R: Rng + SeedableRng + Send + Sync,
        F: Fn(&mut Ensemble) -> Option::<Energy> + Copy + Send + Sync,
        C: Fn() -> bool + Copy + Send + Sync,
        Energy: Sync + Send + Clone,
        walker::RewlWalker<R, Hist, Energy, S, Res>: Send,
    {
        Self::try_greedy_choose_rng_build(self, energy_fn, condition)
    }

    pub fn try_greedy_choose_rng_build<R, R2, F, C, Energy>(self, energy_fn: F, condition: C) -> Result<Rewl<Ensemble, R, Hist, Energy, S, Res>, Self>
    where Hist: HistogramVal<Energy>,
        R: Rng + SeedableRng + Send + Sync,
        Ensemble: HasRng<R2> + Sized,
        R2: Rng + SeedableRng,
        F: Fn(&mut Ensemble) -> Option::<Energy> + Copy + Send + Sync,
        C: Fn() -> bool + Copy + Send + Sync,
        Energy: Sync + Send + Clone,
        walker::RewlWalker<R, Hist, Energy, S, Res>: Send,
    {
        let mut container = Vec::with_capacity(self.hists.len());
        let ensembles = self.ensembles;
        let hists = self.hists;
        let step_size = self.step_size;
        let sweep_size = self.sweep_size;
        ensembles.into_par_iter()
            .zip(hists.into_par_iter())
            .map(
                |(mut e, h)|
                {
                    let mut energy = 'outer: loop
                    {
                        for _ in 0..sweep_size.get(){
                            if let Some(energy) = energy_fn(&mut e){
                                break 'outer energy;
                            }
                            e.m_steps_quiet(step_size.get());
                        }
                        if !condition(){
                            return (h, e, None);
                        }
                    };

                    if !h.is_inside(&energy) {
                        let mut distance = h.distance(&energy);

                        let mut steps = Vec::with_capacity(step_size.get());
                        'outer2: loop 
                        {
                            for _ in 0..sweep_size.get()
                            {
                                e.m_steps(step_size.get(), &mut steps);
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
                                        break 'outer2;
                                    }
                                }else {
                                    e.undo_steps_quiet(&mut steps);
                                }
                            }
                            if !condition()
                            {
                                return (h, e, None);
                            }
                        }
                    }
                    (h, e, Some(energy))
                }
            ).collect_into_vec(&mut container);

        Self::build(
            container,
            self.chunk_size,
            self.log_f_threshold,
            step_size,
            sweep_size
        )
    }

    pub fn try_interval_heuristik_build<R, R2, F, C, Energy>
    (
        self,
        energy_fn: F,
        condition: C,
        overlap: usize
    ) -> Result<Rewl<Ensemble, R, Hist, Energy, S, Res>, Self>
    where Hist: HistogramVal<Energy> + HistogramIntervalDistance<Energy>,
        R: Rng + SeedableRng + Send + Sync,
        Ensemble: HasRng<R> + Sized,
        F: Fn(&mut Ensemble) -> Option::<Energy> + Copy + Send + Sync,
        C: Fn() -> bool + Copy + Send + Sync,
        Energy: Sync + Send + Clone,
        walker::RewlWalker<R, Hist, Energy, S, Res>: Send,
    {
        Self::try_interval_heuristik_choose_rng_build(self, energy_fn, condition, overlap)
    }

    pub fn try_interval_heuristik_choose_rng_build<R, R2, F, C, Energy>
    (
        self,
        energy_fn: F,
        condition: C,
        overlap: usize
    ) -> Result<Rewl<Ensemble, R, Hist, Energy, S, Res>, Self>
    where Hist: HistogramVal<Energy> + HistogramIntervalDistance<Energy>,
        R: Rng + SeedableRng + Send + Sync,
        Ensemble: HasRng<R2> + Sized,
        R2: Rng + SeedableRng,
        F: Fn(&mut Ensemble) -> Option::<Energy> + Copy + Send + Sync,
        C: Fn() -> bool + Copy + Send + Sync,
        Energy: Sync + Send + Clone,
        walker::RewlWalker<R, Hist, Energy, S, Res>: Send,
    {
        let mut container = Vec::with_capacity(self.hists.len());
        let ensembles = self.ensembles;
        let hists = self.hists;
        let step_size = self.step_size;
        let sweep_size = self.sweep_size;
        ensembles.into_par_iter()
            .zip(hists.into_par_iter())
            .map(
                |(mut e, h)|
                {
                    let mut energy = 'outer: loop
                    {
                        for _ in 0..sweep_size.get(){
                            if let Some(energy) = energy_fn(&mut e){
                                break 'outer energy;
                            }
                            e.m_steps_quiet(step_size.get());
                        }
                        if !condition(){
                            return (h, e, None);
                        }
                    };

                    if !h.is_inside(&energy) {
                        let mut distance = h.interval_distance_overlap(&energy, overlap);

                        let mut steps = Vec::with_capacity(step_size.get());
                        'outer2: loop 
                        {
                            for _ in 0..sweep_size.get()
                            {
                                e.m_steps(step_size.get(), &mut steps);
                                let current_energy = if let Some(energy) = energy_fn(&mut e)
                                {
                                    energy
                                } else {
                                    e.undo_steps_quiet(&mut steps);
                                    continue;
                                };
    
                                let new_distance = h.interval_distance_overlap(&current_energy, overlap);
                                if new_distance <= distance {
                                    energy = current_energy;
                                    distance = new_distance;
                                    if distance == 0 {
                                        break 'outer2;
                                    }
                                }else {
                                    e.undo_steps_quiet(&mut steps);
                                }
                            }
                            if !condition()
                            {
                                return (h, e, None);
                            }
                        }
                    }
                    (h, e, Some(energy))
                }
            ).collect_into_vec(&mut container);

        Self::build(
            container,
            self.chunk_size,
            self.log_f_threshold,
            step_size,
            sweep_size
        )
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
            .all(|w| w.log_f() < self.log_f_threshold)
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
            .map(|w| w.hist())
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
            .map(|w| w.id())
            .collect()
    }
}
