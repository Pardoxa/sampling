use crate::*;
use crate::rewl::*;
use rand::{Rng, SeedableRng, Error};
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

    pub fn interval_heuristik_build<R, R2, F, Energy>
    (
        self,
        energy_fn: F,
        overlap: usize
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


    pub fn try_interval_heuristik_build<R, F, C, Energy>
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
    
    pub fn interval_heuristik_choose_rng_build<R, R2, F, Energy>
    (
        self,
        energy_fn: F,
        overlap: usize
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

    pub fn mixed_heuristik_build<R, F, Energy>
    (
        self,
        energy_fn: F,
        overlap: usize
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

    pub fn try_mixed_heuristik_build<R, F, C, Energy>
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
        Self::try_mixed_heuristik_choose_rng_build(self, energy_fn, condition, overlap)
    }


    pub fn try_mixed_heuristik_choose_rng_build<R, R2, F, C, Energy>
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
                        let mut distance_interval;
                        let mut distance;

                        let mut steps = Vec::with_capacity(step_size.get());
                        'outer2: loop 
                        {
                            distance_interval = h.interval_distance_overlap(&energy, overlap);
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
}