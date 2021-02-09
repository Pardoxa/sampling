use crate::*;
use crate::rewl::*;
use crate::glue_helper::*;
use rand::{Rng, SeedableRng, prelude::SliceRandom};
use std::{num::NonZeroUsize, sync::*};
use rayon::prelude::*;

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

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
