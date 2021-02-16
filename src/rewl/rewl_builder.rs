use crate::*;
use crate::rewl::*;
use rand::{Rng, SeedableRng, Error};
use std::{marker::PhantomData, num::NonZeroUsize, sync::*};
use rayon::prelude::*;

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

/// # Use this to create a replica exchange wang landau simulation
/// * Tipp: Use shorthand `RewlBuilder`
/// ## Notes
/// * Don't be intimidated by the number of trait bounds an generic parameters.
/// You should very rarely have to explicitly write the types, as Rust will infer them for you.
#[derive(Debug,Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct ReplicaExchangeWangLandauBuilder<Ensemble, Hist, S, Res>
{
    walker_per_interval: NonZeroUsize,
    ensembles: Vec<Ensemble>,
    hists: Vec<Hist>,
    finished: Vec<bool>,
    log_f_threshold: f64,
    sweep_size: NonZeroUsize,
    step_size: Vec<usize>,
    res: PhantomData<Res>,
    s: PhantomData<S>
}

/// # Short for `ReplicaExchangeWangLandauBuilder`
pub type RewlBuilder<Ensemble, Hist, S, Res> = ReplicaExchangeWangLandauBuilder<Ensemble, Hist, S, Res>;

/// # Errors
/// that can arise during the construction of RewlBuilder
#[derive(Debug)]
pub enum RewlBuilderErr{
    /// * The threshold for `log_f` needs to be a normal number.
    /// * That basically means: the number is neither zero, infinite, subnormal, or NaN. 
    /// For more info, see the [Documentation](`std::primitive::f64::is_normal`)
    NonNormalThreshold,
    /// log_f_threshold must not be negative
    Negative,
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

    /// # Fraction of finished intervals
    /// * which fraction of the intervals has found valid starting configurations?
    /// ## Note
    /// * even if every interval has a valid configuration directly after using one of 
    /// the `from_…` methods, it fill show a fraction of 0.0 - the fraction 
    /// will only be correct after calling one of the `…build` methods (on the Error of the result)
    pub fn finished_fraction(&self) -> f64
    {
        let done = self.finished
            .iter()
            .filter(|&&f| f)
            .count();
        
        done as f64 / self.finished.len() as f64
    }

    /// # Is the interval in a valid statring configuration?
    /// Check which intervals have valid starting points
    /// ## Note
    /// * in the beginning the RewlBuilder has no way of knowing, if the intervals have
    /// valid starting configuration - as it does not know the energy function yet.
    /// Therefore this will only be correct after calling one of the `…build` methods 
    /// (on the Error of the result)
    pub fn finished_slice(&self) -> &[bool]
    {
        &self.finished
    }

    /// # Read access to histograms
    pub fn hists(&self) -> &[Hist]
    {
        &self.hists
    }

    /// # Read access to the ensembles
    pub fn ensembles(&self) -> &[Ensemble]
    {
        &self.ensembles
    }

    /// # Change step size of individual intervals
    /// * change step size of intervals
    pub fn step_size(&mut self) -> &mut [usize]
    {
        &mut self.step_size
    }

    /// # new rewl builder
    /// * used to create a **R**eplica **e**xchange **w**ang **l**andau simulation.
    /// * use this method, if you want to have fine control over each walker, i.e., if you can
    /// provide ensembles, who's energy is already inside the corresponding intervals `hists`
    /// * you might want to use [from_ensemble](crate::ReplicaExchangeWangLandauBuilder::from_ensemble) or
    /// [from_ensemble_tuple](crate::ReplicaExchangeWangLandauBuilder::from_ensemble_tuple) instead
    ///
    /// | Parameter             | meaning                                                                                                                                                  |
    /// |-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
    /// | `ensembles`           | a vector of ensembles, one for each interval. Corresponds to the `hists` entries.                                                                        |
    /// | `hists`               | Overlapping intervals for the wang landau walkers. Should be sorted according to their respective left bins.                                             |
    /// | `step_size`           | step_size for the markov steps, which will be performed                                                                                                  |
    /// | `sweep_size`          | How many steps will be performed until the replica exchanges are proposed                                                                                |
    /// | `walker_per_interval` | How many walkers should be used for each interval (entry of `hists`)                                                                                     |
    /// | `log_f_threshold`     | Threshold for the logaritm of the factor f (see paper). Rewl Simulation is finished, when all(!) walkers have a factor log_f smaller than this threshold |
    ///
    /// ## Notes
    /// * for proper statistics, you should seed the random number generators (used for the markov chain) of all ensembles 
    /// differently!
    /// * `log_f_threshold` has to be a [normal](`std::primitive::f64::is_normal`) and non negative number
    /// * each enty of `ensembles` will be cloned `walker_per_interval - 1` times and their respective rngs will be 
    /// seeded via the `HasRng` trait
    pub fn from_ensemble_vec(
        ensembles: Vec<Ensemble>,
        hists: Vec<Hist>,
        step_size: usize,
        sweep_size: NonZeroUsize,
        walker_per_interval: NonZeroUsize,
        log_f_threshold: f64
    ) -> Result<Self, RewlBuilderErr>
    {
        if !log_f_threshold.is_normal(){
            return Err(RewlBuilderErr::NonNormalThreshold);
        }
        if log_f_threshold < 0.0 {
            return Err(RewlBuilderErr::Negative);
        }
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
        let step_size = (0..ensembles.len())
            .map(|_| step_size)
            .collect();

        let finished = vec![false; hists.len()];
        Ok(
            Self{
                ensembles,
                step_size,
                sweep_size,
                walker_per_interval,
                hists,
                log_f_threshold,
                s: PhantomData::<S>,
                res: PhantomData::<Res>,
                finished
            }
        )
    }
    
    /// # Create a builder to create a replica exchange wang landau (Rewl) simulation
    /// * creates vector of ensembles and (re)seeds their respective rngs (by using the `HasRng` trait)
    /// * calls [`Self::from_ensemble_vec(…)`](`crate::ReplicaExchangeWangLandauBuilder::from_ensemble_vec`) afterwards,
    /// look there for more information about the parameter
    pub fn from_ensemble<R>(
        ensemble: Ensemble,
        hists: Vec<Hist>,
        step_size: usize,
        sweep_size: NonZeroUsize,
        walker_per_interval: NonZeroUsize,
        log_f_threshold: f64,

    ) -> Result<Self,RewlBuilderErr>
    where Ensemble: HasRng<R> + Clone,
        R: Rng + SeedableRng
    {
        let len = NonZeroUsize::new(hists.len())
            .ok_or(RewlBuilderErr::Empty)?;
        let ensembles = Self::clone_and_seed_ensembles(ensemble, len)?;
        Self::from_ensemble_vec(ensembles, hists, step_size, sweep_size, walker_per_interval, log_f_threshold)
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

    /// # Create a builder to create a replica exchange wang landau (Rewl) simulation
    /// * creates vector of ensembles and (re)seeds their respective rngs (by using the `HasRng` trait).
    /// The vector is created by cloning `ensemble_tuple.0` for everything up to the middle of the vector and 
    /// `ensemble_tuple.1` for the rest. The length of the vector will be the same as `hists.len()`.
    /// If It is an uneven number, the middle element will be a clone of `ensemble_tuple.1`
    /// * calls [`Self::from_ensemble_vec(…)`](`crate::ReplicaExchangeWangLandauBuilder::from_ensemble_vec`) afterwards,
    /// look there for more information about the parameter
    /// * use this, if you know configurations, that would be good starting points for finding 
    /// configurations at either end of the intervals. 
    pub fn from_ensemble_tuple<R>(
        ensemble_tuple: (Ensemble, Ensemble),
        hists: Vec<Hist>,
        step_size: usize,
        sweep_size: NonZeroUsize,
        walker_per_interval: NonZeroUsize,
        log_f_threshold: f64,
    ) -> Result<Self,RewlBuilderErr>
    where Ensemble: HasRng<R> + Clone,
        R: Rng + SeedableRng
    {
        let len = NonZeroUsize::new(hists.len())
            .ok_or(RewlBuilderErr::Empty)?;
        if len < unsafe{NonZeroUsize::new_unchecked(2)} {
            return Err(RewlBuilderErr::LenMissmatch);
        }
        let (left, mut right) = ensemble_tuple;
        let mut ensembles = Vec::with_capacity(len.get());
        let mid = len.get() / 2;

        for _ in 1..mid {
            let mut e = left.clone();
            let mut rng = R::from_rng(right.rng())
               .map_err(|e| RewlBuilderErr::SeedError(e))?;
            e.swap_rng(&mut rng);
            ensembles.push(e);
        }
        ensembles.push(left);
        for _ in mid..len.get()-1
        {
            let mut e = right.clone();
            let mut rng = R::from_rng(right.rng())
               .map_err(|e| RewlBuilderErr::SeedError(e))?;
            e.swap_rng(&mut rng);
            ensembles.push(e);
        }
        ensembles.push(right);

        Self::from_ensemble_vec(ensembles, hists, step_size, sweep_size, walker_per_interval, log_f_threshold)
    }

    fn build<Energy, R, R2>
    (
        container: Vec<(Hist, Ensemble, Option<Energy>)>,
        walker_per_interval: NonZeroUsize,
        log_f_threshold: f64,
        step_size: Vec<usize>,
        sweep_size: NonZeroUsize,
        finished: Vec<bool>

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
                    walker_per_interval,
                    s: PhantomData::<S>,
                    res: PhantomData::<Res>,
                    log_f_threshold,
                    step_size,
                    sweep_size,
                    finished
                }
            );
        }

        let mut ensembles_rw_lock = Vec::with_capacity(container.len() * walker_per_interval.get());
        let mut walker = Vec::with_capacity(walker_per_interval.get() * container.len());
        let mut counter = 0;

        for ((mut h, mut e, energy), step_size) in container.into_iter()
            .zip(step_size.into_iter())
        {
            let energy = energy.unwrap();
            h.reset();
            for _ in 0..walker_per_interval.get()-1 {
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
                        step_size,
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
                    step_size,
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
                chunk_size: walker_per_interval,
                walker,
                log_f_threshold
            }
        )
    }


    /// # Create `Rewl`, i.e., Replica exchange wang landau simulation
    /// * uses a greedy heuristik to find valid configurations, meaning configurations that 
    /// are within the required intervals, i.e., histograms
    /// ## Note
    /// * Depending on how complex your energy landscape is, this can take a very long time,
    /// maybe not even terminating at all.
    /// * You can use `self.try_greedy_choose_rng_build` to limit the time of the search
    pub fn greedy_build<R, F, Energy>(self, energy_fn: F) -> Rewl<Ensemble, R, Hist, Energy, S, Res>
    where Hist: HistogramVal<Energy>,
        Ensemble: HasRng<R> + Sized,
        R: Rng + SeedableRng + Send + Sync,
        F: Fn(&mut Ensemble) -> Option::<Energy> + Copy + Send + Sync,
        Energy: Sync + Send + Clone,
        walker::RewlWalker<R, Hist, Energy, S, Res>: Send,
    {
        match Self::try_greedy_choose_rng_build(self, energy_fn, || true){
            Ok(result) => result,
            _ => unreachable!()
        }
    }
    
    /// # Create `Rewl`, i.e., Replica exchange wang landau simulation
    /// * similar to [`greedy_build`](`crate::ReplicaExchangeWangLandauBuilder::greedy_build`)
    /// * `condition` can be used to limit the time of the search - it will end when `condition`
    /// returns false.
    /// ##Note
    /// * condition will only be checked once every sweep, i.e., every `sweep_size` markov steps
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

    /// # Create `Rewl`, i.e., Replica exchange wang landau simulation
    /// * similar to [`greedy_build`](`crate::ReplicaExchangeWangLandauBuilder::greedy_build`)
    /// * Difference: You can choose a different `Rng` for the Wang Landau walkers (i.e., the
    /// acceptance of the replica exchange moves etc.)
    /// * usage: `self.greedy_choose_rng_build::<RNG,_,_,_>(energy_fn)`
    pub fn greedy_choose_rng_build<R, R2, F, Energy>(self, energy_fn: F) -> Rewl<Ensemble, R, Hist, Energy, S, Res>
    where Hist: HistogramVal<Energy>,
        R: Rng + SeedableRng + Send + Sync,
        Ensemble: HasRng<R2> + Sized,
        R2: Rng + SeedableRng,
        F: Fn(&mut Ensemble) -> Option::<Energy> + Copy + Send + Sync,
        Energy: Sync + Send + Clone,
        walker::RewlWalker<R, Hist, Energy, S, Res>: Send,
    {
        match self.try_greedy_choose_rng_build(energy_fn, || true)
        {
            Ok(result) => result,
            _ => unreachable!()
        }
    }

    /// # Create `Rewl`, i.e., Replica exchange wang landau simulation
    /// * similar to [`try_greedy_build`](`crate::ReplicaExchangeWangLandauBuilder::try_greedy_build`)
    /// * Difference: You can choose a different `Rng` for the Wang Landau walkers (i.e., the
    /// acceptance of the replica exchange moves etc.)
    /// * usage: `self.try_greedy_choose_rng_build::<RNG,_,_,_,_>(energy_fn, condition)`
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
        let mut finished = self.finished;
        ensembles.into_par_iter()
            .zip(hists.into_par_iter())
            .zip(step_size.par_iter())
            .zip(finished.par_iter_mut())
            .map(
                |(((mut e, h), &step_size), finished)|
                {
                    let mut energy = 'outer: loop
                    {
                        for _ in 0..sweep_size.get(){
                            if let Some(energy) = energy_fn(&mut e){
                                break 'outer energy;
                            }
                            e.m_steps_quiet(step_size);
                        }
                        if !condition(){
                            return (h, e, None);
                        }
                    };

                    if !h.is_inside(&energy) {
                        let mut distance = h.distance(&energy);

                        let mut steps = Vec::with_capacity(step_size);
                        'outer2: loop 
                        {
                            for _ in 0..sweep_size.get()
                            {
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
                    *finished = true;
                    (h, e, Some(energy))
                }
            ).collect_into_vec(&mut container);

        Self::build(
            container,
            self.walker_per_interval,
            self.log_f_threshold,
            step_size,
            sweep_size,
            finished
        )
    }

    /// # Create `Rewl`, i.e., Replica exchange wang landau simulation
    /// * uses an interval heuristik to find valid configurations, meaning configurations that 
    /// are within the required intervals, i.e., histograms
    /// * Uses overlapping intervals. Accepts a step, if the resulting ensemble is in the same interval as before,
    /// or it is in an interval closer to the target interval. 
    /// Take a look at the [`HistogramIntervalDistance` trait](`crate::HistogramIntervalDistance`)
    /// * `overlap` should smaller than the number of bins in your histogram. E.g. `overlap = 3` if you have 200 bins
    /// 
    /// ## Note
    /// * Depending on how complex your energy landscape is, this can take a very long time,
    /// maybe not even terminating at all.
    /// * You can use [`try_interval_heuristik_build`](`crate::ReplicaExchangeWangLandauBuilder::try_interval_heuristik_build`) to limit the time of the search
    pub fn interval_heuristik_build<R, R2, F, Energy>
    (
        self,
        energy_fn: F,
        overlap: NonZeroUsize
    ) -> Rewl<Ensemble, R, Hist, Energy, S, Res>
    where Hist: HistogramVal<Energy> + HistogramIntervalDistance<Energy>,
        R: Rng + SeedableRng + Send + Sync,
        Ensemble: HasRng<R> + Sized,
        F: Fn(&mut Ensemble) -> Option::<Energy> + Copy + Send + Sync,
        Energy: Sync + Send + Clone,
        walker::RewlWalker<R, Hist, Energy, S, Res>: Send,
    {
        match Self::try_interval_heuristik_build(self, energy_fn, || true, overlap){
            Ok(result) => result,
            _ => unreachable!()
        }
    }


    /// # Create `Rewl`, i.e., Replica exchange wang landau simulation
    /// * similar to [`interval_heuristik_build`](`crate::ReplicaExchangeWangLandauBuilder::interval_heuristik_build`)
    /// * `condition` can be used to limit the time of the search - it will end when `condition`
    /// returns false.
    /// ##Note
    /// * condition will only be checked once every sweep, i.e., every `sweep_size` markov steps
    pub fn try_interval_heuristik_build<R, F, C, Energy>
    (
        self,
        energy_fn: F,
        condition: C,
        overlap: NonZeroUsize
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
    
    /// # Create `Rewl`, i.e., Replica exchange wang landau simulation
    /// * similar to [`try_interval_heuristik_build`](`crate::ReplicaExchangeWangLandauBuilder::try_interval_heuristik_build`)
    /// * Difference: You can choose a different `Rng` for the Wang Landau walkers (i.e., the
    /// acceptance of the replica exchange moves etc.)
    /// * usage: `self.try_interval_heuristik_build::<RNG,_,_,_,_>(energy_fn, overlap)`
    pub fn interval_heuristik_choose_rng_build<R, R2, F, Energy>
    (
        self,
        energy_fn: F,
        overlap: NonZeroUsize
    ) -> Rewl<Ensemble, R, Hist, Energy, S, Res>
    where Hist: HistogramVal<Energy> + HistogramIntervalDistance<Energy>,
        R: Rng + SeedableRng + Send + Sync,
        Ensemble: HasRng<R2> + Sized,
        R2: Rng + SeedableRng,
        F: Fn(&mut Ensemble) -> Option::<Energy> + Copy + Send + Sync,
        Energy: Sync + Send + Clone,
        walker::RewlWalker<R, Hist, Energy, S, Res>: Send,
    {
        match self.try_interval_heuristik_choose_rng_build(energy_fn, || true, overlap) {
            Ok(result) => result,
            _ => unreachable!()
        }
    }

    /// # Create `Rewl`, i.e., Replica exchange wang landau simulation
    /// * similar to [`interval_heuristik_choose_rng_build`](`crate::ReplicaExchangeWangLandauBuilder::interval_heuristik_choose_rng_build`)
    /// * Difference: You can choose the Random numbver generator used for the Rewl Walkers, i.e., for 
    /// accepting or rejecting the markov steps and replica exchanges. 
    /// * usage: `self.try_interval_heuristik_choose_rng_build<RNG, _,_,_,_>(energy_fn, condition, overlap)]
    pub fn try_interval_heuristik_choose_rng_build<R, R2, F, C, Energy>
    (
        self,
        energy_fn: F,
        condition: C,
        overlap: NonZeroUsize
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
        let mut finished = self.finished;
        ensembles.into_par_iter()
            .zip(hists.into_par_iter())
            .zip(step_size.par_iter())
            .zip(finished.par_iter_mut())
            .map(
                |(((mut e, h), &step_size), finished)|
                {
                    let mut energy = 'outer: loop
                    {
                        for _ in 0..sweep_size.get(){
                            if let Some(energy) = energy_fn(&mut e){
                                break 'outer energy;
                            }
                            e.m_steps_quiet(step_size);
                        }
                        if !condition(){
                            return (h, e, None);
                        }
                    };

                    if !h.is_inside(&energy) {
                        let mut distance = h.interval_distance_overlap(&energy, overlap);

                        let mut steps = Vec::with_capacity(step_size);
                        'outer2: loop 
                        {
                            for _ in 0..sweep_size.get()
                            {
                                e.m_steps(step_size, &mut steps);
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
                    *finished = true;
                    (h, e, Some(energy))
                }
            ).collect_into_vec(&mut container);

        Self::build(
            container,
            self.walker_per_interval,
            self.log_f_threshold,
            step_size,
            sweep_size,
            finished
        )
    }

    /// # Create `Rewl`, i.e., Replica exchange wang landau simulation
    /// * alternates between interval-heuristik and greedy-heuristik
    /// * The interval heuristik uses overlapping intervals. Accepts a step, if the resulting ensemble is in the same interval as before,
    /// or it is in an interval closer to the target interval. 
    /// Take a look at the [`HistogramIntervalDistance` trait](`crate::HistogramIntervalDistance`)
    /// * `overlap` should smaller than the number of bins in your histogram. E.g. `overlap = 3` if you have 200 bins
    /// 
    /// ## Note
    /// * Depending on how complex your energy landscape is, this can take a very long time,
    /// maybe not even terminating at all.
    /// * You can use [`try_mixed_heuristik_build`](`crate::ReplicaExchangeWangLandauBuilder::try_mixed_heuristik_build`) to limit the time of the search
    pub fn mixed_heuristik_build<R, F, Energy>
    (
        self,
        energy_fn: F,
        overlap: NonZeroUsize
    ) -> Rewl<Ensemble, R, Hist, Energy, S, Res>
    where Hist: HistogramVal<Energy> + HistogramIntervalDistance<Energy>,
        R: Rng + SeedableRng + Send + Sync,
        Ensemble: HasRng<R> + Sized,
        F: Fn(&mut Ensemble) -> Option::<Energy> + Copy + Send + Sync,
        Energy: Sync + Send + Clone,
        walker::RewlWalker<R, Hist, Energy, S, Res>: Send,
    {
        match Self::try_mixed_heuristik_choose_rng_build(self, energy_fn, || true, overlap){
            Ok(result) => result,
            Err(_) => unreachable!()
        }
    }

    /// # Create `Rewl`, i.e., Replica exchange wang landau simulation
    /// * alternates between interval-heuristik and greedy-heuristik
    /// * The interval heuristik uses overlapping intervals. Accepts a step, if the resulting ensemble is in the same interval as before,
    /// or it is in an interval closer to the target interval. 
    /// Take a look at the [`HistogramIntervalDistance` trait](`crate::HistogramIntervalDistance`)
    /// * `overlap` should smaller than the number of bins in your histogram. E.g. `overlap = 3` if you have 200 bins
    /// 
    /// ## Note
    /// * `condition` can be used to limit the time of the search - it will end when `condition`
    /// returns false (or a valid solution is found)
    /// * condition will only be checked once every sweep, i.e., every `sweep_size` markov steps
    pub fn try_mixed_heuristik_build<R, F, C, Energy>
    (
        self,
        energy_fn: F,
        condition: C,
        overlap: NonZeroUsize
    ) -> Result<Rewl<Ensemble, R, Hist, Energy, S, Res>, Self>
    where Hist: HistogramVal<Energy> + HistogramIntervalDistance<Energy>,
        R: Rng + SeedableRng + Send + Sync,
        Ensemble: HasRng<R> + Sized,
        F: Fn(&mut Ensemble) -> Option::<Energy> + Copy + Send + Sync,
        C: Fn() -> bool + Copy + Send + Sync,
        Energy: Sync + Send + Clone,
        walker::RewlWalker<R, Hist, Energy, S, Res>: Send,
    {
        Self::try_mixed_heuristik_choose_rng_build(self, energy_fn, condition, overlap)
    }

    /// # Create `Rewl`, i.e., Replica exchange wang landau simulation
    /// * similar to [`try_mixed_heuristik_build`](`crate::ReplicaExchangeWangLandauBuilder::try_mixed_heuristik_build`)
    /// * difference: Lets you choose the rng type for the Rewl simulation, i.e., the rng used for 
    /// accepting or rejecting markov steps and replica exchange moves
    /// * usage: `self.try_mixed_heuristik_choose_rng_build<RNG_TYPE, _, _, _, _>(energy_fn, condition, overlap)`
    pub fn try_mixed_heuristik_choose_rng_build<R, R2, F, C, Energy>
    (
        self,
        energy_fn: F,
        condition: C,
        overlap: NonZeroUsize
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
        let mut finished = self.finished;
        ensembles.into_par_iter()
            .zip(hists.into_par_iter())
            .zip(step_size.par_iter())
            .zip(finished.par_iter_mut())
            .map(
                |(((mut e, h), &step_size), finished)|
                {
                    let mut energy = 'outer: loop
                    {
                        for _ in 0..sweep_size.get(){
                            if let Some(energy) = energy_fn(&mut e){
                                break 'outer energy;
                            }
                            e.m_steps_quiet(step_size);
                        }
                        if !condition(){
                            return (h, e, None);
                        }
                    };

                    if !h.is_inside(&energy) {
                        let mut distance_interval;
                        let mut distance;

                        let mut steps = Vec::with_capacity(step_size);
                        'outer2: loop 
                        {
                            distance_interval = h.interval_distance_overlap(&energy, overlap);
                            for _ in 0..sweep_size.get()
                            {
                                e.m_steps(step_size, &mut steps);
                                let current_energy = if let Some(energy) = energy_fn(&mut e)
                                {
                                    energy
                                } else {
                                    e.undo_steps_quiet(&mut steps);
                                    continue;
                                };
    
                                let new_distance = h.interval_distance_overlap(&current_energy, overlap);
                                if new_distance <= distance_interval {
                                    energy = current_energy;
                                    distance_interval = new_distance;
                                    if distance_interval == 0 {
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
                            distance = h.distance(&energy);
                            for _ in 0..sweep_size.get()
                            {
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
                    *finished = true;
                    (h, e, Some(energy))
                }
            ).collect_into_vec(&mut container);

        Self::build(
            container,
            self.walker_per_interval,
            self.log_f_threshold,
            step_size,
            sweep_size,
            finished
        )
    }
}