use{
    crate::{
        *,
        rewl::*,
        glue_helper::*
    },
    rayon::{iter::ParallelIterator, prelude::*},
    rand::{Rng, SeedableRng, prelude::SliceRandom},
    std::{num::NonZeroUsize, sync::*, cmp::*}
};

#[cfg(feature = "sweep_time_optimization")]
use std::cmp::Reverse;

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};


/// Result of glueing. See [Glued]
pub type GluedResult<Hist, Energy> = Result<Glued<Hist, Energy>, HistErrors>;

/// # Efficient replica exchange Wang landau
/// * use this to quickly build your own parallel replica exchange wang landau simulation
/// ## Tipp
/// Use the short hand [`Rewl`](crate::Rewl)  
/// ## Citations
/// * the following paper were used to progamm this - you should cite them, if you use 
/// this library for a publication!
/// 
/// > Y. W. Li, T. Vogel, T. Wüst and D. P. Landau,
/// > “A new paradigm for petascale Monte Carlo simulation: Replica exchange Wang-Landau sampling,”
/// > J.&nbsp;Phys.: Conf.&nbsp;Ser. **510** 012012 (2014), DOI&nbsp;[10.1088/1742-6596/510/1/012012](https://doi.org/10.1088/1742-6596/510/1/012012)
///
/// > T. Vogel, Y. W. Li, T. Wüst and D. P. Landau,
/// > “Exploring new frontiers in statistical physics with a new, parallel Wang-Landau framework,”
/// > J.&nbsp;Phys.: Conf.&nbsp;Ser. **487** 012001 (2014), DOI&nbsp;[10.1088/1742-6596/487/1/012001](https://doi.org/10.1088/1742-6596/487/1/012001)
///
/// > T. Vogel, Y. W. Li, T. Wüst and D. P. Landau,
/// > “Scalable replica-exchange framework for Wang-Landau sampling,”
/// > Phys.&nbsp;Rev.&nbsp;E **90**: 023302 (2014), DOI&nbsp;[10.1103/PhysRevE.90.023302](https://doi.org/10.1103/PhysRevE.90.023302)
///
/// > R. E. Belardinelli and V. D. Pereyra,
/// > “Fast algorithm to calculate density of states,”
/// > Phys.&nbsp;Rev.&nbsp;E&nbsp;**75**: 046701 (2007), DOI&nbsp;[10.1103/PhysRevE.75.046701](https://doi.org/10.1103/PhysRevE.75.046701)
/// 
/// > F. Wang and D. P. Landau,
/// > “Efficient, multiple-range random walk algorithm to calculate the density of states,” 
/// > Phys.&nbsp;Rev.&nbsp;Lett.&nbsp;**86**, 2050–2053 (2001), DOI&nbsp;[10.1103/PhysRevLett.86.2050](https://doi.org/10.1103/PhysRevLett.86.2050)
#[derive(Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct ReplicaExchangeWangLandau<Ensemble, R, Hist, Energy, S, Res>{
    pub(crate) chunk_size: NonZeroUsize,
    pub(crate) ensembles: Vec<RwLock<Ensemble>>,
    pub(crate) walker: Vec<RewlWalker<R, Hist, Energy, S, Res>>,
    pub(crate) log_f_threshold: f64,
    pub(crate) replica_exchange_mode: bool,
    pub(crate) roundtrip_halfes: Vec<usize>,
    pub(crate) last_extreme_interval_visited: Vec<ExtremeInterval>
}

impl<Ensemble, R, Hist, Energy, S, Res> GlueAble<Hist> for ReplicaExchangeWangLandau<Ensemble, R, Hist, Energy, S, Res>
where Hist: Clone
{

    fn push_glue_entry_ignoring(
        &self, 
        job: &mut GlueJob<Hist>,
        ignore_idx: &[usize]
    ) {

        job.round_trips
            .extend(self.roundtrip_iter());

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
                    let mut progress = f64::NEG_INFINITY;
                    let mut accepted = 0;
                    let mut rejected = 0;
                    let mut replica_exchanges = 0_u64;
                    let mut proposed_replica_exchanges = 0;
                    for w in walker{
                        let log_f = w.log_f();
                        if log_f > progress {
                            progress = log_f;
                        }
                        let r = w.rejected_markov_steps();
                        let a = w.step_count() - r;
                        rejected += r;
                        accepted += a;
                        replica_exchanges += w.replica_exchanges() as u64;
                        proposed_replica_exchanges += w.proposed_replica_exchanges();
                    }

                    let stats = IntervalSimStats{
                        sim_progress: SimProgress::LogF(progress),
                        interval_sim_type: SimulationType::REWL,
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

/// # Enum used internally
/// It will save if the corresponding interval is the leftest one, the rightes one
/// or none of that
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum ExtremeInterval
{
    /// There is no interval that is "more left" then this one
    Left,
    /// There is no interval that is "more right" then this one
    Right,
    /// None of the above
    None
}

/// # Error types for threshold log_f
/// This enum gives you information on the error during the 
/// setting of a threshold
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum ThresholdErrors{
    /// No negative threshold value allowed
    Negative,
    /// The threshold cannot be subnormal
    NonNormal,
    /// The threshold is not allowed to be zero
    Zero,
}

/// Short for [`ReplicaExchangeWangLandau`](crate::rewl::ReplicaExchangeWangLandau), 
/// which you can look at for citations
pub type Rewl<Ensemble, R, Hist, Energy, S, Res> = ReplicaExchangeWangLandau<Ensemble, R, Hist, Energy, S, Res>;


impl<Ensemble, R, Hist, Energy, S, Res> Rewl<Ensemble, R, Hist, Energy, S, Res>
{
    /// # Read access to internal rewl walkers
    /// * each of these walkers independently samples an interval. 
    /// * see paper for more infos
    pub fn walkers(&self) -> &Vec<RewlWalker<R, Hist, Energy, S, Res>>
    {
        &self.walker
    }


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
    /// * `None` if `index` out of range
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

    /// # mut access to your ensembles
    /// * if possible, prefer [`get_ensemble`](Self::get_ensemble)
    /// * None if `index` out of range
    /// ## Safety
    /// * it is assumed, that whatever you change has no effect on the 
    /// Markov Chain, the result of the energy function etc. 
    /// * might **panic** if a thread is poisened
    pub unsafe fn get_ensemble_mut(&mut self, index: usize) -> Option<&mut Ensemble>
    {
        self.ensembles
            .get_mut(index)
            .map(|e| e.get_mut().unwrap())
    }

    /// # Get the number of intervals present
    pub fn num_intervals(&self) -> NonZeroUsize
    {
        match NonZeroUsize::new(self.walker.len() / self.chunk_size.get())
        {
            Some(v) => v,
            None => unreachable!()
        }
    }

    /// Returns number of walkers per interval
    pub fn walkers_per_interval(&self) -> NonZeroUsize
    {
        self.chunk_size
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
                        "Fatal Error encountered; ERRORCODE 0x9 - \
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
                        "Fatal Error encountered; ERRORCODE 0xA - \
                        Sweep sizes of intervals do not match! \
                        This should be impossible! if you are using the latest version of the \
                        'sampling' library, please contact the library author via github by opening an \
                        issue! https://github.com/Pardoxa/sampling/issues"
                    )
                );
            Some(sweep_size)
        }
    }

    fn get_log_prob_and_hists(&self) -> (Vec<&Hist>, Vec<Vec<f64>>)
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

    /// # Minimum of roundtrips
    ///
    /// Definition of roundtrip:
    /// If a walker is in the leftest interval, then in the rightest and then in the leftest again 
    /// (or the other way around) then this is counted as one roundtrip.
    /// 
    /// This will return the minimum of roundtrips
    pub fn min_roundtrips(&self) -> usize 
    {
        match self.roundtrip_iter().min()
        {
            Some(v) => v,
            None => unreachable!()
        }
    }

    /// # Maximum of roundtrips
    ///
    /// Definition of roundtrip:
    /// If a walker is in the leftest interval, then in the rightest and then in the leftest again 
    /// (or the other way around) then this is counted as one roundtrip.
    /// 
    /// This will return the maximum of roundtrips
    pub fn max_roundtrips(&self) -> usize 
    {
        match self.roundtrip_iter().max()
        {
            Some(v) => v,
            None => unreachable!()
        }
    }

    #[inline]
    /// # Roundtrips
    /// Definition of roundtrip:
    /// If a walker is in the leftest interval, then in the rightest and then in the leftest again 
    /// (or the other way around) then this is counted as one roundtrip.
    /// 
    /// This will return an iterator over the roundtrips
    pub fn roundtrip_iter(&'_ self) -> impl Iterator<Item=usize> + '_
    {
        self.roundtrip_halfes
            .iter()
            .map(|&r_h| r_h / 2)
    }

    /// returns largest value of factor log_f present in the walkers
    pub fn largest_log_f(&self) -> f64
    {
        self.walker
            .iter()
            .map(|w| w.log_f())
            .fold(std::f64::NEG_INFINITY, |acc, x| x.max(acc))

    }

    /// # Log_f factors of the walkers
    /// * the log_f's will be reduced towards 0 during the simulation
    pub fn log_f_vec(&self) -> Vec<f64>
    {
        self.walker
            .iter()
            .map(|w| w.log_f())
            .collect()
    }

    /// # change the threshold of log_f
    /// * it has to be a positive, normal number
    pub fn set_log_f_threshold(&mut self, new_threshold: f64) -> Result<f64, ThresholdErrors>
    {
        if !new_threshold.is_normal()
        {   
            Err(ThresholdErrors::NonNormal)
        } else if new_threshold < 0.0 {
            Err(ThresholdErrors::Negative)
        } else if new_threshold == 0.0 {
            Err(ThresholdErrors::Zero)
        } else{
            let old_threshold = self.log_f_threshold;
            self.log_f_threshold = new_threshold;
            Ok(old_threshold)
        }
    }

    /// # Is the simulation finished?
    /// checks if **all** walkers have factors `log_f`
    /// that are below the threshold you chose
    pub fn is_finished(&self) -> bool
    {
        self.walker
            .iter()
            .all(|w| w.log_f() < self.log_f_threshold)
    }

    /// # Results of the simulation
    /// 
    /// This is what we do the simulation for!
    /// 
    /// It uses derivative merging to give you a [Glued] which you can use to write
    /// the data into a file.
    /// The derivative merged is explained in [derivative_merged_log_prob_and_aligned](crate::rees::ReplicaExchangeEntropicSampling::derivative_merged_log_prob_and_aligned)
    ///
    /// ## Notes
    /// Fails if the internal histograms (intervals) do not align. Might fail if 
    /// there is no overlap between neighboring intervals 
    pub fn derivative_merged_log_prob_and_aligned(&self) -> Result<Glued<Hist, Energy>, HistErrors>
    where Hist: HistogramCombine + Histogram
    {
        let (hists, log_probs) = self.get_log_prob_and_hists();
        derivative_merged_and_aligned(
            log_probs, &hists, LogBase::BaseE
        )
    }

    /// # Results of the simulation
    /// 
    /// This is what we do the simulation for!
    /// 
    /// It uses average merging to give you a [Glued] which you can use to write
    /// the data into a file.
    /// The average merged is explained in  [average_merged_and_aligned](crate::glue::average_merged_and_aligned)
    ///
    /// ## Notes
    /// Fails if the internal histograms (intervals) do not align. Might fail if 
    /// there is no overlap between neighboring intervals 
    pub fn average_merged_log_probability_and_align(&self)-> Result<Glued<Hist, Energy>, HistErrors>
    where Hist: HistogramCombine + Histogram
    {
        let (hists, log_probs) = self.get_log_prob_and_hists();
        average_merged_and_aligned(
            log_probs, 
            &hists, 
            LogBase::BaseE
        )
    }        

    /// # Get Ids
    /// This is an indicator that the replica exchange works.
    /// In the beginning, this will be a sorted vector, e.g. \[0,1,2,3,4\].
    /// Then it will show, where the ensemble, which the corresponding walkers currently work with,
    /// originated from. E.g. If the vector is \[3,1,0,2,4\], Then walker 0 has a
    /// ensemble originating from walker 3, the walker 1 is back to its original 
    /// ensemble, walker 2 has an ensemble originating form walker 0 and so on.
    pub fn get_id_vec(&self) -> Vec<usize>
    {
        self.walker
            .iter()
            .map(|w| w.id())
            .collect()
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

    /// # Convert into Rees
    /// This creates a Replica exchange entropic sampling simulation 
    /// from this Replica exchange wang landau simulation
    pub fn into_rees(self) -> Rees<(), Ensemble, R, Hist, Energy, S, Res>
    where Hist: Histogram
    {
        self.into()
    }

    /// # Convert into Rees
    /// * similar to [into_rees](`crate::rewl::Rewl::into_rees`), though now we can store extra information.
    /// The extra information can be anything, e.g., files in which 
    /// each walker should later write information every nth step or something 
    /// else entirely.
    /// 
    /// # important
    /// * The vector `extra` must be exactly as long as the walker slice and 
    /// each walker is assigned the corresponding entry from the vector `extra`
    /// * You can look at the walker slice with the [walkers](`crate::rewl::Rewl::walkers`) method
    #[allow(clippy::type_complexity, clippy::result_large_err)]
    pub fn into_rees_with_extra<Extra>(self, extra: Vec<Extra>) -> Result<Rees<Extra, Ensemble, R, Hist, Energy, S, Res>, (Self, Vec<Extra>)>
    where Hist: Histogram
    {
        if extra.len() != self.walker.len()
        {
            Err((self, extra))
        } else {
            let rewl_roundtrips: Vec<_> = self.roundtrip_iter().collect();
            let rees_roundtrip_halfes: Vec<_> = vec![0; rewl_roundtrips.len()];
            let rees_last_extreme_interval_visited: Vec<_> = vec![ExtremeInterval::None; rewl_roundtrips.len()];

            let mut walker = Vec::with_capacity(self.walker.len());
            walker.extend(
                self.walker
                    .into_iter()
                    .map(|w| w.into())
            );

            let mut rees = 
            Rees{
                walker,
                ensembles: self.ensembles,
                replica_exchange_mode: self.replica_exchange_mode,
                extra,
                chunk_size: self.chunk_size,
                rewl_roundtrips,
                rees_last_extreme_interval_visited,
                rees_roundtrip_halfes
            };
            rees.update_roundtrips();
            Ok(
                rees
            )
            
        }
    }
}


impl<Ensemble, R, Hist, Energy, S, Res> Rewl<Ensemble, R, Hist, Energy, S, Res> 
where R: Send + Sync + Rng + SeedableRng,
    Hist: Send + Sync + Histogram + HistogramVal<Energy>,
    Energy: Send + Sync + Clone,
    Ensemble: MarkovChain<S, Res>,
    Res: Send + Sync,
    S: Send + Sync
{


    /// # Perform the Replica exchange wang landau simulation
    /// * will simulate until **all** walkers have factors `log_f`
    /// that are below the threshold you chose
    pub fn simulate_until_convergence<F>(
        &mut self,
        energy_fn: F
    )
    where 
        Ensemble: Send + Sync,
        R: Send + Sync,
        F: Fn(&mut Ensemble) -> Option<Energy> + Copy + Send + Sync
    {
        while !self.is_finished()
        {
            self.sweep(energy_fn);
        }
    }

    /// # Perform the Replica exchange wang landau simulation
    /// * will simulate until **all** walkers have factors `log_f`
    /// that are below the threshold you chose **or**
    /// * until condition returns false
    pub fn simulate_while<F, C>(
        &mut self,
        energy_fn: F,
        mut condition: C
    )
    where 
        Ensemble: Send + Sync,
        R: Send + Sync,
        F: Fn(&mut Ensemble) -> Option<Energy> + Copy + Send + Sync,
        C: FnMut(&Self) -> bool
    {
        while !self.is_finished() && condition(self)
        {
            self.sweep(energy_fn);
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
    Ensemble: Sync + Send
    {
        let ensembles = self.ensembles.as_slice();
        self.walker
            .par_iter()
            .all(|w| w.check_energy_fn(ensembles, energy_fn))
    }

    /// # Sweep
    /// * Performs one sweep of the Replica exchange wang landau simulation
    /// * You can make a complete simulation, by repeatatly calling this method
    /// until `self.is_finished()` returns true
    pub fn sweep<F>(&mut self, energy_fn: F)
    where Ensemble: Send + Sync,
        R: Send + Sync,
        F: Fn(&mut Ensemble) -> Option<Energy> + Copy + Send + Sync
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
            .for_each(|w| w.wang_landau_sweep(slice, energy_fn));

        
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

        self.update_roundtrips();
    }

    
    pub(crate) fn update_roundtrips(&mut self){
        if self.num_intervals().get() == 1 {
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
                    let last_visited = match self.last_extreme_interval_visited.get_mut(id){
                        Some(last) => last,
                        None => unreachable!()
                    };

                    match last_visited {
                        ExtremeInterval::Right => {
                            *last_visited = ExtremeInterval::Left;
                            self.roundtrip_halfes[id] += 1;
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
                    let last_visited = match self.last_extreme_interval_visited.get_mut(id){
                        Some(last) => last,
                        None => unreachable!()
                    };

                    match last_visited {
                        ExtremeInterval::Left => {
                            *last_visited = ExtremeInterval::Right;
                            self.roundtrip_halfes[id] += 1;
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

/// # Merge probability density of multiple rewl simulations
/// * Will calculate the merged log (base 10) probability density. Also returns the corresponding histogram.
/// * If an interval has multiple walkers, their probability will be merged before all probabilities are aligned
/// * `rewls` does not need to be sorted in any way
/// ## Errors
/// * will return `HistErrors::EmptySlice` if the `rees` slice is empty
/// * will return other HistErrors if the intervals have no overlap
pub fn merged_log10_prob<Ensemble, R, Hist, Energy, S, Res>(rewls: &[Rewl<Ensemble, R, Hist, Energy, S, Res>]) -> Result<(Vec<f64>, Hist), HistErrors>
where Hist: Histogram + HistogramVal<Energy> + HistogramCombine + Send + Sync,
    Energy: PartialOrd
{
    let mut res = merged_log_prob(rewls)?;
    ln_to_log10(&mut res.0);
    Ok(res)
}

/// # Merge probability density of multiple rewl simulations
/// * Will calculate the merged log (base e) probability density. Also returns the corresponding histogram.
/// * If an interval has multiple walkers, their probability will be merged before all probabilities are aligned
/// * `rewls` does not need to be sorted in any way
/// ## Errors
/// * will return `HistErrors::EmptySlice` if the `rees` slice is empty
/// * will return other HistErrors if the intervals have no overlap
pub fn merged_log_prob<Ensemble, R, Hist, Energy, S, Res>(rewls: &[Rewl<Ensemble, R, Hist, Energy, S, Res>]) -> Result<(Vec<f64>, Hist), HistErrors>
where Hist: Histogram + HistogramVal<Energy> + HistogramCombine + Send + Sync,
    Energy: PartialOrd
{
    if rewls.is_empty() {
        return Err(HistErrors::EmptySlice);
    }
    let merged_prob = merged_probs(rewls);
    let container = combine_container(rewls, &merged_prob, true);
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


/// Helper to ignore specific intervals/walkers
pub(crate) fn ignore_fn<T>(container: &mut Vec<T>, ignore: &[usize])
{
    let mut ignore = ignore.to_vec();
    // sorting in reverse, to remove correct indices later on
    ignore.sort_unstable_by_key(|&e| Reverse(e));
    // remove duplicates
    ignore.dedup();
    // remove indices
    ignore.into_iter()
        .for_each(
            |i|
            {
                if i < container.len(){
                    let _ = container.remove(i);
                }
            }
        );
}


fn merged_probs<Ensemble, R, Hist, Energy, S, Res>
(
    rewls: &[Rewl<Ensemble, R, Hist, Energy, S, Res>]
) -> Vec<Vec<f64>>
{
    let merged_probs: Vec<_> = rewls.iter()
        .flat_map(
            |rewl|
            {
                rewl.walkers()
                    .chunks(rewl.walkers_per_interval().get())
                    .map(get_merged_walker_prob)
            }
        ).collect();
    merged_probs
}

fn combine_container<'a, Ensemble, R, Hist, Energy, S, Res>
(
    rewls: &'a [Rewl<Ensemble, R, Hist, Energy, S, Res>],
    log_probabilities: &'a [Vec<f64>],
    merged: bool
) ->  Vec<(&'a [f64], &'a Hist)>
where Hist: HistogramVal<Energy> + HistogramCombine,
    Energy: PartialOrd
{
    let mut step_by = NonZeroUsize::new(1).unwrap();
    let hists: Vec<_> = rewls.iter()
        .flat_map(
            |rewl|
            {
                if merged {
                    step_by = rewl.walkers_per_interval();
                }
                rewl.walkers()
                    .iter()
                    .step_by(step_by.get())
                    .map(|w| w.hist())
            }
        ).collect();

    assert_eq!(hists.len(), log_probabilities.len());

    let mut container: Vec<_> = log_probabilities
        .iter()
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
