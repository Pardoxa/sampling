use{
    crate::{
        *,
        rees::replica_exchange,
        rewl::{Rewl, ignore_fn},
        glue_helper::{subtract_max, ln_to_log10},
        glue::derivative::*
    },
    rand::{Rng, prelude::SliceRandom},
    std::{num::NonZeroUsize, sync::*, cmp::*},
    rayon::prelude::*
};

#[cfg(feature = "sweep_time_optimization")]
use std::cmp::Reverse;

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

/// # Struct used for entropic sampling with replica exchanges
/// See [this](crate::rees), also for merge functions to create the 
/// final probability density functions
#[derive(Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct ReplicaExchangeEntropicSampling<Extra, Ensemble, R, Hist, Energy, S, Res>
{
    pub(crate) chunk_size: NonZeroUsize,
    pub(crate) ensembles: Vec<RwLock<Ensemble>>,
    pub(crate) walker: Vec<ReesWalker<R, Hist, Energy, S, Res>>,
    pub(crate) replica_exchange_mode: bool,
    pub(crate) extra: Vec<Extra>,
    pub(crate) rewl_roundtrips: Vec<usize>,
    pub(crate) rees_roundtrip_halfes: Vec<usize>,
    pub(crate) rees_last_extreme_interval_visited: Vec<ExtremeInterval>
}

impl<Extra, Ensemble, R, Hist, Energy, S, Res> GlueAble<Hist> for ReplicaExchangeEntropicSampling<Extra, Ensemble, R, Hist, Energy, S, Res>
where Hist: Clone + Histogram
{

    fn push_glue_entry_ignoring(
        &self, 
        job: &mut GlueJob<Hist>,
        ignore_idx: &[usize]
    ) {
        job.round_trips
            .extend(self.rees_roundtrip_iter());

        let (hists, probs) = self.get_log_prob_and_hists();

        self.walker
            .chunks(self.chunk_size.get())
            .zip(hists)
            .zip(probs)
            .enumerate()
            .filter_map(|(index, ((walker, hist), prob))|
                {
                    if ignore_idx.contains(&index){
                        None
                    } else {
                        Some(((walker, hist), prob))
                    }
                }
            )
            .for_each(
                |((walker, hist), prob)|
                {
                    let mut missing_steps = 0;
                    let mut accepted = 0;
                    let mut rejected = 0;
                    let mut replica_exchanges = 0;
                    let mut proposed_replica_exchanges = 0;
                    for w in walker{
                        
                        if !w.is_finished(){
                            let missing = w.step_threshold() - w.step_count();
                            if missing > missing_steps{
                                missing_steps = missing;
                            }
                        }

                        let r = w.rejected_markov_steps();
                        let a = w.step_count() - r;
                        rejected += r;
                        accepted += a;
                        replica_exchanges += w.replica_exchanges();
                        proposed_replica_exchanges += w.proposed_replica_exchanges();
                    }

                    let stats = IntervalSimStats{
                        sim_progress: SimProgress::MissingSteps(missing_steps),
                        interval_sim_type: SimulationType::REES,
                        rejected_steps: rejected,
                        accepted_steps: accepted,
                        replica_exchanges: Some(replica_exchanges),
                        proposed_replica_exchanges: Some(proposed_replica_exchanges),
                        merged_over_walkers: self.chunk_size
                    };

                    job.collection.push(
                        GlueEntry{ 
                            hist: hist.clone(), 
                            prob, 
                            log_base: LogBase::BaseE, 
                            interval_stats: stats
                        }
                    );
                }
            )
    }
}

/// # Short for [ReplicaExchangeEntropicSampling]
pub type Rees<Extra, Ensemble, R, Hist, Energy, S, Res> = ReplicaExchangeEntropicSampling<Extra, Ensemble, R, Hist, Energy, S, Res>;

impl<Ensemble, R, Hist, Energy, S, Res, Extra>  Rees<Extra, Ensemble, R, Hist, Energy, S, Res>
{

    /// # Iterator over ensembles
    /// If you do not know what `RwLockReadGuard<'a, Ensemble>` is - do not worry.
    /// you can just pretend it is `&Ensemble` and everything should work out fine,
    /// since it implements [`Deref`](https://doc.rust-lang.org/std/ops/trait.Deref.html).
    /// Of cause, you can also take a look at [`RwLockReadGuard`](https://doc.rust-lang.org/std/sync/struct.RwLockReadGuard.html)
    pub fn ensemble_iter(&'_ self) -> impl Iterator<Item=RwLockReadGuard<'_, Ensemble>>
    {
        self.ensembles
            .iter()
            .map(|e| e.read().unwrap())
    }

    /// # read access to your ensembles
    /// * None if index out of range
    /// * If you do not know what `RwLockReadGuard<Ensemble>` is - do not worry.
    /// you can just pretend it is `&Ensemble` and everything will work out fine,
    /// since it implements [`Deref`](https://doc.rust-lang.org/std/ops/trait.Deref.html).
    /// Of cause, you can also take a look at [`RwLockReadGuard`](https://doc.rust-lang.org/std/sync/struct.RwLockReadGuard.html)
    pub fn get_ensemble(&self, index: usize) -> Option<RwLockReadGuard<Ensemble>>
    {
        self.ensembles
            .get(index)
            .map(|e| e.read().unwrap())
    }

    /// # mut access to your ensembles
    /// * if possible, prefer [`get_ensemble`](Self::get_ensemble)
    /// * **unsafe** only use this if you know what you are doing
    /// * `None` if `index` out of range
    /// ## Safety
    /// * might **panic** if a thread is poisened
    /// * it is assumed, that whatever you change has no effect on the 
    /// Markov Chain, the result of the energy function etc. 
    pub unsafe fn get_ensemble_mut(&mut self, index: usize) -> Option<&mut Ensemble>
    {
        self.ensembles
            .get_mut(index)
            .map(|e| e.get_mut().unwrap())
    }

    /// # Mutable iterator over ensembles
    /// * if possible, prefer [`ensemble_iter`](Self::ensemble_iter)
    /// ## Safety
    /// * it is assumed, that whatever you change has no effect on the 
    /// Markov Chain, the result of the energy function etc. 
    /// * might **panic** if a thread is poisened
    pub unsafe fn ensemble_iter_mut(&mut self) -> impl Iterator<Item=&mut Ensemble>
    {
        self.ensembles
            .iter_mut()
            .map(|item| item.get_mut().unwrap())
    }

    /// # read access to the internal histograms used by the walkers
    pub fn hists(&self) -> Vec<&Hist>
    {
        self.walker.iter()
            .map(|w| w.hist())
            .collect()
    }

    /// # read access to internal histogram
    /// * None if index out of range
    pub fn get_hist(&self, index: usize) -> Option<&Hist>
    {
        self.walker
            .get(index)
            .map(|w| w.hist())
    }

    fn get_log_prob_and_hists(&self) -> (Vec<&Hist>, Vec<Vec<f64>>)
    where Hist: Histogram
    {
            // get the log_probabilities - the walkers over the same intervals are merged
            let log_prob: Vec<_> = self.walker
                .chunks(self.chunk_size.get())
                .map(get_merged_walker_prob)
                .collect();
            
            let hists: Vec<_> = self.walker.iter()
                .step_by(self.chunk_size.get())
                .map(|w| w.hist())
                .collect();
            (hists, log_prob)
    }

    /// # Iterate over the roundtrips done by the REWL
    /// This returns an Iterator which returns the number of roundtrips for each walker.
    /// Roundtrips are defined as follows:
    /// 
    /// If a walker is in the leftest interval, then in the rightest and then in the leftest again 
    /// (or the other way around) then this is counted as one roundtrip.
    /// Note: If only one interval exists, no roundtrips are possible
    /// 
    /// This iterator will return the roundtrips from the REWL simulation
    pub fn rewl_roundtrip_iter(&'_ self) -> impl Iterator<Item=usize> + '_
    {
        self.rewl_roundtrips
            .iter()
            .copied()
    }

    /// # Iterator over roundtrips done by REES
    /// - same as [rewl_roundtrip_iter](Self::rewl_roundtrip_iter) just for the rees roundtrips
    pub fn rees_roundtrip_iter(&'_ self) -> impl Iterator<Item=usize> + '_
    {
        self.rees_roundtrip_halfes
            .iter()
            .map(|half| half / 2)
    }
}

impl<Ensemble, R, Hist, Energy, S, Res> From<Rewl<Ensemble, R, Hist, Energy, S, Res>> for Rees<(), Ensemble, R, Hist, Energy, S, Res>
where Hist: Histogram
{
    fn from(rewl: Rewl<Ensemble, R, Hist, Energy, S, Res>) -> Self 
    {
        let extra = vec![(); rewl.walker.len()];

        let rees_result = rewl.into_rees_with_extra(extra);

        match rees_result{
            Ok(rees) => rees,
            Err(_) => unreachable!()
        }
    }
}

impl<Extra, Ensemble, R, Hist, Energy, S, Res> Rees<Extra, Ensemble, R, Hist, Energy, S, Res>
{
    /// # Checks threshold
    /// returns true, if all walkers are [finished](`crate::rees::ReesWalker::is_finished`)
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

    /// # Returns internal walkers
    /// * access to internal slice of walkers
    /// * the walkers are sorted and neighboring walker are either 
    /// sampling the same interval, or a neighboring 
    /// (and if the replica exchange makes any sense overlapping)
    /// interval
    pub fn walkers(&self) -> &[ReesWalker<R, Hist, Energy, S, Res>]
    {
        &self.walker
    }

    /// # Change step size for markov chain of walkers
    /// * changes the step size used in the sweep
    /// * changes step size of all walkers in the nth interval
    /// * returns Err if index out of bounds, i.e., the requested interval does not exist
    /// * interval counting starts at 0, i.e., n=0 is the first interval
    #[allow(clippy::result_unit_err)]
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

    /// # Get step size for markov chain of walkers
    /// * returns `None` if index out of bounds, i.e., the requested interval does not exist
    /// * interval counting starts at 0, i.e., n=0 is the first interval
    pub fn get_step_size_of_interval(&self, n: usize) -> Option<usize>
    {
        let start = n * self.chunk_size.get();
        let end = start + self.chunk_size.get();

        if self.walker.len() < end {
            None
        } else {
            let slice = &self.walker[start..start+self.chunk_size.get()];
            let step_size = slice[0].step_size();
            slice[1..]
                .iter()
                .for_each(|w| 
                    assert_eq!(
                        step_size, w.step_size(), 
                        "Fatal Error encountered; ERRORCODE 0x8 - \
                        Sweep sizes of intervals do not match! \
                        This should be impossible! if you are using the latest version of the \
                        'sampling' library, please contact the library author via github by opening an \
                        issue! https://github.com/Pardoxa/sampling/issues"
                    )
                );
            Some(step_size)
        }
    }

    /// # Change sweep size for markov chain of walkers
    /// * changes the sweep size used in the sweep
    /// * changes sweep size of all walkers in the nth interval
    /// * returns Err if index out of bounds, i.e., the requested interval does not exist
    /// * interval counting starts at 0, i.e., n=0 is the first interval
    #[allow(clippy::result_unit_err)]
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

    /// # Get sweep size for markov chain of walkers
    /// * returns `None` if index out of bounds, i.e., the requested interval does not exist
    /// * interval counting starts at 0, i.e., n=0 is the first interval
    pub fn get_sweep_size_of_interval(&self, n: usize) -> Option<NonZeroUsize>
    {
        let start = n * self.chunk_size.get();
        let end = start + self.chunk_size.get();

        if self.walker.len() < end {
            None
        } else {
            let slice = &self.walker[start..start+self.chunk_size.get()];
            let sweep_size = slice[0].sweep_size();
            slice[1..]
                .iter()
                .for_each(|w| 
                    assert_eq!(
                        sweep_size, w.sweep_size(), 
                        "Fatal Error encountered; ERRORCODE 0x2 - \
                        Sweep sizes of intervals do not match! \
                        This should be impossible! if you are using the latest version of the \
                        'sampling' library, please contact the library author via github by opening an \
                        issue! https://github.com/Pardoxa/sampling/issues"
                    )
                );
            Some(sweep_size)
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
    #[allow(clippy::type_complexity)]
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
            rees_last_extreme_interval_visited: self.rees_last_extreme_interval_visited,
            rees_roundtrip_halfes: self.rees_roundtrip_halfes,
            rewl_roundtrips: self.rewl_roundtrips
        };
        (
            rees,
            old_extra
        )
    }

    /// # Swap the extra vector
    /// * Note: len of extra has to be the same as `self.num_walkers()` (which is the same as `self.extra_slice().len()`)
    /// otherwise an Err is returned
    #[allow(clippy::result_unit_err, clippy::type_complexity)]
    pub fn swap_extra<Extra2>(
        self, 
        new_extra: Vec<Extra2>
    ) -> Result<(Rees<Extra2, Ensemble, R, Hist, Energy, S, Res>, Vec<Extra>), ()>
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
                rees_last_extreme_interval_visited: self.rees_last_extreme_interval_visited,
                rees_roundtrip_halfes: self.rees_roundtrip_halfes,
                rewl_roundtrips: self.rewl_roundtrips
            };
            Ok(
                (
                    rees,
                    old_extra
                )
            )
        }
    }

    pub(crate) fn update_roundtrips(&mut self)
    {
        if self.num_intervals() <= 1 {
            return;
        }
        // check all walker that are currently in the first interval
        let mut chunk_iter = self.walker.chunks(self.chunk_size.get());
        let first_chunk = chunk_iter.next().unwrap();
        first_chunk.iter()
            .for_each(
                |walker|
                {
                    let id = walker.id();
                    let last_visited = match self.rees_last_extreme_interval_visited.get_mut(id){
                        Some(last) => last,
                        None => unreachable!()
                    };

                    match last_visited {
                        ExtremeInterval::Right => {
                            *last_visited = ExtremeInterval::Left;
                            self.rees_roundtrip_halfes[id] += 1;
                        },
                        ExtremeInterval::None => {
                            *last_visited = ExtremeInterval::Left;
                        },
                        _ => ()
                    }
                }
            );

        // check all walker that are currently in the last interval
        let last_chunk = match chunk_iter.last()
        {
            Some(chunk) => chunk,
            None => unreachable!()
        };

        last_chunk.iter()
            .for_each(
                |walker|
                {
                    let id = walker.id();
                    let last_visited = match self.rees_last_extreme_interval_visited.get_mut(id){
                        Some(last) => last,
                        None => unreachable!()
                    };

                    match last_visited {
                        ExtremeInterval::Left => {
                            *last_visited = ExtremeInterval::Right;
                            self.rees_roundtrip_halfes[id] += 1;
                        },
                        ExtremeInterval::None => {
                            *last_visited = ExtremeInterval::Right;
                        },
                        _ => ()
                    }
                }
            );
    }
}

impl<Ensemble, R, Hist, Energy, S, Res> Rees<(), Ensemble, R, Hist, Energy, S, Res>
{
    /// # Add extra information to your Replica Exchange entropic sampling simulation
    /// * can be used to, e.g., print stuff during the simulation, or write it to a file and so on
    #[allow(clippy::type_complexity, clippy::result_large_err)]
    pub fn add_extra<Extra>(
        self, 
        extra: Vec<Extra>
    ) -> Result<Rees<Extra, Ensemble, R, Hist, Energy, S, Res>, (Self, Vec<Extra>)>
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
                rees_last_extreme_interval_visited: self.rees_last_extreme_interval_visited,
                rees_roundtrip_halfes: self.rees_roundtrip_halfes,
                rewl_roundtrips: self.rewl_roundtrips
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

    /// # Refine the estimate of the probability density functions
    /// * refines the estimate of all walkers
    /// * does so by calling the walker method [refine](`crate::rees::ReesWalker::refine`)
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
        P: Fn(&ReesWalker<R, Hist, Energy, S, Res>, &mut Ensemble, &mut Extra) + Copy + Send + Sync,
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
            );
        
        self.update_roundtrips()
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
        P: Fn(&ReesWalker<R, Hist, Energy, S, Res>, &mut Ensemble, &mut Extra) + Copy + Send + Sync,
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
        P: Fn(&ReesWalker<R, Hist, Energy, S, Res>, &mut Ensemble, &mut Extra) + Copy + Send + Sync,
    {
        while !self.is_finished() && condition(self)
        {
            self.sweep(energy_fn, extra_fn);
        }
    }

    /// # Sanity check
    /// * checks if the stored (i.e., last) energy(s) of the system
    /// match with the result of energy_fn
    pub fn check_energy_fn<F>(
        &mut self,
        energy_fn: F
    )   -> bool
    where Energy: PartialEq,
    F: Fn(&mut Ensemble) -> Option<Energy> + Copy + Send + Sync,
    {
        let ensembles = self.ensembles.as_slice();
        self.walker
            .par_iter()
            .all(|w| w.check_energy_fn(ensembles, energy_fn))
    }

    #[allow(clippy::type_complexity)]
    #[deprecated]
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
            .for_each(|v| 
                {
                    subtract_max(v);
                }
            );


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
    /// Fails if the internal histograms (intervals) do not align. Might fail if 
    /// there is no overlap between neighboring intervals 
    #[deprecated(since="0.2.0", note="will be removed in future releases. Use new method 'derivative_merged_log_prob_and_aligned' or consider using 'average_merged_log_probability_and_align' instead")]
    #[allow(deprecated)]
    pub fn merged_log_prob_rees(&self) -> Result<(Hist, Vec<f64>), HistErrors>
    where Hist: HistogramCombine
    {
        let (mut log_prob, e_hist) = self.merged_log_probability_rees()?;

        norm_ln_prob(&mut log_prob);
        
        Ok((e_hist, log_prob))
    }

    fn get_glue_stats(&self) -> GlueStats
    {
        let stats = self.walker
            .chunks(self.chunk_size.get())
            .map(
                |walker|
                {
                    let mut missing_steps = 0;
                    let mut accepted = 0;
                    let mut rejected = 0;
                    let mut replica_exchanges = 0;
                    let mut proposed_replica_exchanges = 0;
                    for w in walker{
                        
                        if !w.is_finished(){
                            let missing = w.step_threshold() - w.step_count();
                            if missing > missing_steps{
                                missing_steps = missing;
                            }
                        }

                        let r = w.rejected_markov_steps();
                        let a = w.step_count() - r;
                        rejected += r;
                        accepted += a;
                        replica_exchanges += w.replica_exchanges();
                        proposed_replica_exchanges += w.proposed_replica_exchanges();
                    }

                    IntervalSimStats{
                        sim_progress: SimProgress::MissingSteps(missing_steps),
                        interval_sim_type: SimulationType::REES,
                        rejected_steps: rejected,
                        accepted_steps: accepted,
                        replica_exchanges: Some(replica_exchanges),
                        proposed_replica_exchanges: Some(proposed_replica_exchanges),
                        merged_over_walkers: self.chunk_size
                    }
                }
            ).collect();
        let roundtrips = self.rees_roundtrip_iter().collect();
        GlueStats { interval_stats: stats, roundtrips }
    }

    /// # Results of the simulation
    /// 
    /// This is what we do the simulation for!
    /// 
    /// It returns [Glued] which allows you to print out the merged probability density function.
    /// It also allows you to switch the base of the logarithm and so on, have a look!
    /// 
    /// It will use an average based merging algorithm, i.e., it will try to align the intervals
    /// and merge them by using the values obtained by averaging in log-space 
    /// 
    /// ## Notes
    /// Fails if the internal histograms (intervals) do not align. Might fail if 
    /// there is no overlap between neighboring intervals
    pub fn average_merged_log_probability_and_align(&self)-> Result<Glued<Hist, Energy>, HistErrors>
    where Hist: HistogramCombine
    {
        let (hists, log_probs) = self.get_log_prob_and_hists();

        let mut res = average_merged_and_aligned(
            log_probs, &hists, LogBase::BaseE
        )?;
        let stats = self.get_glue_stats();
        res.set_stats(stats);
        Ok(res)
    }

    /// # Results of the simulation
    /// 
    /// This is what we do the simulation for!
    /// 
    /// It returns [Glued] which allows you to print out the merged probability density function.
    /// It also allows you to switch the base of the logarithm and so on, have a look!
    /// 
    /// It will use an derivative based merging algorithm, i.e., it will try to align the intervals
    /// and merge them by looking at the derivatives of the probability density function.
    /// It will search for the (merging-)point where the derivatives are the most similar to each other 
    /// and glue by using the values of one of the intervals before the merging point and the 
    /// other interval afterwards. This is repeated for every interval
    /// 
    /// ## Notes
    /// Fails if the internal histograms (intervals) do not align. Might fail if 
    /// there is no overlap between neighboring intervals
    pub fn derivative_merged_log_prob_and_aligned(&self) -> Result<Glued<Hist, Energy>, HistErrors>
    where Hist: HistogramCombine + Histogram
    {
        let (hists, log_probs) = self.get_log_prob_and_hists();
        
        let mut res = derivative_merged_and_aligned(log_probs, &hists, LogBase::BaseE)?;
        let stats = self.get_glue_stats();
        res.set_stats(stats);
        Ok(res)
    }

    #[deprecated(since="0.2.0", note="will be removed in future releases. Use new method 'derivative_merged_log_prob_and_aligned' or consider using 'average_merged_log_probability_and_align' instead")]
    #[allow(deprecated)]
    fn merged_log_probability_rees(&self) -> Result<(Vec<f64>, Hist), HistErrors>
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


}

/// # Merge probability density of multiple rees simulations
/// * Will calculate the merged log (base e) probability density. Also returns the corresponding histogram.
/// * `rees` does not need to be sorted in any way
/// ## Errors
/// * will return `HistErrors::EmptySlice` if the `rees` slice is empty
/// * will return other HistErrors if the intervals have no overlap
pub fn merged_log_prob_rees<Extra, Ensemble, R, Hist, Energy, S, Res>(
    rees: &[Rees<Extra, Ensemble, R, Hist, Energy, S, Res>]
) -> Result<(Vec<f64>, Hist), HistErrors>
where Hist: Histogram + HistogramVal<Energy> + HistogramCombine + Send + Sync,
    Energy: PartialOrd
{
    merged_log_prob_ignore(rees, &[])
}

/// # Merge probability density of multiple rees simulations
/// * very similar to [merged_log_prob](`crate::rees::Rees::merged_log_prob`)
/// 
/// The difference is, that this function will ignore the specified walkers,
/// therefore `ignore` should be a slice of indices, which are to be ignored.
/// The slice does not have to be sorted in any way, though duplicate indices 
/// and indices which are out of bounds will be ignored for the ignore list
pub fn merged_log_prob_ignore<Extra, Ensemble, R, Hist, Energy, S, Res>(
    rees: &[Rees<Extra, Ensemble, R, Hist, Energy, S, Res>],
    ignore: &[usize]
) -> Result<(Vec<f64>, Hist), HistErrors>
where Hist: HistogramVal<Energy> + HistogramCombine + Histogram + Send + Sync,
    Energy: PartialOrd
{
    if rees.is_empty() {
        return Err(HistErrors::EmptySlice);
    }
    let merged_prob = merged_probs(rees);
    let mut container = combine_container(rees, &merged_prob);
    ignore_fn(&mut container, ignore);
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

/// # Merge probability density of multiple rees simulations
/// * Will calculate the merged log (base 10) probability density. Also returns the corresponding histogram.
/// * If an interval has multiple walkers, their probability will be merged before all probabilities are aligned
/// * `rees` does not need to be sorted in any way
/// ## Errors
/// * will return `HistErrors::EmptySlice` if the `rees` slice is empty
/// * will return other HistErrors if the intervals have no overlap
pub fn merged_log10_prob_rees<Extra, Ensemble, R, Hist, Energy, S, Res>(
    rees: &[Rees<Extra, Ensemble, R, Hist, Energy, S, Res>]
) -> Result<(Vec<f64>, Hist), HistErrors>
where Hist: Histogram + HistogramVal<Energy> + HistogramCombine + Send + Sync,
    Energy: PartialOrd
{
    let mut res = merged_log_prob_rees(rees)?;
    ln_to_log10(&mut res.0);
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
        .zip(hists)
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
                        .zip(density)
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