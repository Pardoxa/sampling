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

/// Result of glueing
/// * `Hist` is the histogram which shows the corresponding bins,
/// * `Vec<f64>` is the result of the gluing and merging of the individual intervals
/// * `Vec<Vec<f64>>` are the individual intervals, which are ready to be glued, i.e.,
///    their logarithmic "hight" was allready corrected
pub type Glued<Hist> = (Hist, Vec<f64>, Vec<Vec<f64>>);

// I want to do an even bigger refactoring
pub struct FormerGlued<Hist>
{
    encapsulating_hist: Hist,
    glued: Vec<f64>,
    aligned: Vec<Vec<f64>>,
    base: LogBase
}

impl<Hist> FormerGlued<Hist>
{
    pub fn glued(&self) -> &[f64]
    {
        &self.glued
    }

    pub fn alined(&self) -> &[Vec<f64>]
    {
        &self.aligned
    }

    pub fn base(&self) -> LogBase
    {
        self.base
    }

    pub fn encapsulating_hist(&self) -> &Hist
    {
        &self.encapsulating_hist
    }

    /// Change from Base 10 to Base E or the other way round
    pub fn switch_base(&mut self)
    {
        match self.base
        {
            LogBase::Base10 => {
                log10_to_ln(&mut self.glued);
                self.aligned
                    .iter_mut()
                    .for_each(|interval| log10_to_ln( interval));
                self.base = LogBase::BaseE;
            },
            LogBase::BaseE => {
                ln_to_log10(&mut self.glued);
                self.aligned
                    .iter_mut()
                    .for_each(|interval| ln_to_log10( interval));
                self.base = LogBase::Base10;
            }
        }
    }
}
/// Result of glueing. See [Glued]
pub type GluedResult<Hist> = Result<Glued<Hist>, HistErrors>;

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
    pub(crate) roundtrip_halves: Vec<usize>,
    pub(crate) last_extreme_interval_visited: Vec<ExtremeInterval>
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum ExtremeInterval
{
    Left,
    Right,
    None
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
    /// * **unsafe** only use this if you know what you are doing
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
    /// * **unsafe** only use this if you know what you are doing
    /// * it is assumed, that whatever you change has no effect on the 
    /// Markov Chain, the result of the energy function etc. 
    /// * None if `index` out of range
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
                            self.roundtrip_halves[id] += 1;
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
                            self.roundtrip_halves[id] += 1;
                        },
                        ExtremeInterval::None => {
                            *last_visited = ExtremeInterval::Right;
                        },
                        _ => ()
                    }
                }
            );

    }


    pub fn min_roundtrips(&self) -> usize 
    {
        match self.roundtrip_iter().min()
        {
            Some(v) => v,
            None => unreachable!()
        }
    }

    pub fn max_roundtrips(&self) -> usize 
    {
        match self.roundtrip_iter().max()
        {
            Some(v) => v,
            None => unreachable!()
        }
    }

    #[inline]
    pub fn roundtrip_iter(&'_ self) -> impl Iterator<Item=usize> + '_
    {
        self.roundtrip_halves
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

    /// # Is the simulation finished?
    /// checks if **all** walkers have factors `log_f`
    /// that are below the threshold you chose
    pub fn is_finished(&self) -> bool
    {
        self.walker
            .iter()
            .all(|w| w.log_f() < self.log_f_threshold)
    }

    /// # Result of the simulations!
    /// This is what we do the simulation for!
    /// 
    /// It returns the log10 of the normalized (i.e. sum=1 within numerical precision) probability density and the 
    /// histogram, which contains the corresponding bins.
    ///
    /// Failes if the internal histograms (invervals) do not align. Might fail if 
    /// there is no overlap between neighboring intervals 
    pub fn merged_log10_prob(&self) -> Result<(Hist, Vec<f64>), HistErrors>
    where Hist: HistogramCombine
    {
        let (e_hist, mut log_prob) = self.merged_log_prob()?;

        // switch base of log
        ln_to_log10(&mut log_prob);

        Ok((e_hist, log_prob))

    }

    /// TODO CHeck documentation, changed result type
    /// # Results of the simulation
    /// 
    /// This is what we do the simulation for!
    /// 
    /// It returns histogram, which contains the corresponding bins and
    /// the logarithm with base 10 of the normalized (i.e. sum=1 within numerical precision) 
    /// probability density. Lastly it returns the vector of the aligned probability estimates (also log10) of the
    /// different intervals. This can be used to see, how good the simulation worked,
    /// e.g., by plotting them to see, if they match
    ///
    /// ## Notes
    /// Failes if the internal histograms (invervals) do not align. Might fail if 
    /// there is no overlap between neighboring intervals 
    pub fn merged_log10_prob_and_aligned(&self) -> Result<FormerGlued<Hist>, HistErrors>
    where Hist: HistogramCombine
    {
        let (e_hist, mut log_prob, mut aligned) = self.merged_log_prob_and_aligned()?;
        
        ln_to_log10(&mut log_prob);
        
        aligned.par_iter_mut()
            .for_each(
                |slice| 
                {
                    ln_to_log10(slice);
                }
            );

        let glued = FormerGlued{
            encapsulating_hist: e_hist,
            glued: log_prob,
            aligned,
            base: LogBase::Base10
        };
        Ok(glued)
    }



    /// # Result of the simulations!
    /// This is what we do the simulation for!
    /// 
    /// It returns the natural logarithm of the normalized (i.e. sum=1 within numerical precision) probability density and the 
    /// histogram, which contains the corresponding bins.
    ///
    /// Failes if the internal histograms (invervals) do not align. Might fail if 
    /// there is no overlap between neighboring intervals 
    pub fn merged_log_prob(&self) -> Result<(Hist, Vec<f64>), HistErrors>
    where Hist: HistogramCombine
    {
        let (mut log_prob, e_hist) = self.merged_log_probability()?;

        norm_ln_prob(&mut log_prob);
        
        Ok((e_hist, log_prob))
    }

    /// # Results of the simulation
    /// 
    /// This is what we do the simulation for!
    /// 
    /// It returns histogram, which contains the corresponding bins and
    /// the natural logarithm of the normalized (i.e. sum=1 within numerical precision) 
    /// probability density. Lastly it returns the vector of the aligned probability estimates (also ln) of the
    /// different intervals. This can be used to see, how good the simulation worked,
    /// e.g., by plotting them to see, if they match
    ///
    /// ## Notes
    /// Failes if the internal histograms (invervals) do not align. Might fail if 
    /// there is no overlap between neighboring intervals 
    pub fn merged_log_prob_and_aligned(&self) -> GluedResult<Hist>
    where Hist: HistogramCombine 
    {
        let (e_hist, mut log_prob, mut aligned) = self.merged_log_probability_and_align()?;

        let shift = norm_ln_prob(&mut log_prob);
        
        aligned.par_iter_mut()
            .for_each(
                |aligned|
                {
                    aligned.iter_mut()
                        .for_each(|val| *val -= shift)
                }
            );
        Ok(
            (
                e_hist,
                log_prob,
                aligned
            )
        )
    }

    fn merged_log_probability(&self) -> Result<(Vec<f64>, Hist), HistErrors>
    where Hist: HistogramCombine
    {
        let (hists, log_probs) = self.get_log_prob_and_hists();
        let (merge_points, alignment, log_prob, e_hist) = 
            self.merged_log_probability_helper2(log_probs, hists)?;
        Ok(
            only_merged(
                merge_points,
                alignment,
                log_prob,
                e_hist
            )
        )
    }

    // TODO Rename function
    pub fn no_deriv_merged_log_probability_and_align(&self)-> Result<ReplicaGlued<Hist>, HistErrors>
    where Hist: HistogramCombine
    {
        let (hists, log_probs) = self.get_log_prob_and_hists();
        let (alignment, log_prob, e_hist) = 
            average_merged_log_probability_helper2(log_probs, hists)?;

        Ok(
            no_derive_merged_and_aligned(
                alignment,
                log_prob,
                e_hist
            )
        )
    }

    fn merged_log_probability_and_align(&self) -> GluedResult<Hist>
    where Hist: HistogramCombine
    {
        let (hists, log_probs) = self.get_log_prob_and_hists();
        let (merge_points, alignment, log_prob, e_hist) = 
            self.merged_log_probability_helper2(log_probs, hists)?;
        merged_and_aligned(
            self.walker.iter()
                    .step_by(self.walkers_per_interval().get())
                    .map(|v| v.hist()),
            merge_points,
            alignment,
            log_prob,
            e_hist
        )
    }

    fn get_log_prob_and_hists(&self) -> (Vec<&Hist>, Vec<Vec<f64>>)
    {
        // get the log_probabilities - the walkers over the same intervals are merged
        let log_prob: Vec<_> = self.walker
            .par_chunks(self.chunk_size.get())
            .map(get_merged_walker_prob)
            .collect();
        
        let hists: Vec<_> = self.walker.iter()
            .step_by(self.chunk_size.get())
            .map(|w| w.hist())
            .collect();
        (hists, log_prob)
    }        

    #[allow(clippy::type_complexity)]
    fn merged_log_probability_helper2(
        &self,
        mut log_prob: Vec<Vec<f64>>,
        hists: Vec<&Hist>
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<Vec<f64>>, Hist), HistErrors>
    where Hist: HistogramCombine
    {
        // get the log_probabilities - the walkers over the same intervals are merged
        log_prob
            .par_iter_mut()
            .for_each(
                |v| 
                {
                    subtract_max(v);
                }
            );


        // get the derivative, for merging later
        let derivatives: Vec<_> = log_prob.par_iter()
            .map(|v| derivative_merged(v))
            .collect();

        let e_hist = Hist::encapsulating_hist(&hists)?;

        let alignment  = hists.iter()
            .zip(hists.iter().skip(1))
            .map(|(&left, &right)| left.align(right))
            .collect::<Result<Vec<_>, _>>()?;
        
        
        let merge_points = calc_merge_points(&alignment, &derivatives);

        Ok(
            (
                merge_points,
                alignment,
                log_prob,
                e_hist
            )
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
    #[allow(clippy::type_complexity)]
    pub fn into_rees_with_extra<Extra>(self, extra: Vec<Extra>) -> Result<Rees<Extra, Ensemble, R, Hist, Energy, S, Res>, (Self, Vec<Extra>)>
    {
        if extra.len() != self.walker.len()
        {
            Err((self, extra))
        } else {
            let mut walker = Vec::with_capacity(self.walker.len());
            walker.extend(
                self.walker
                    .into_iter()
                    .map(|w| w.into())
            );
            let rees = 
            Rees{
                walker,
                ensembles: self.ensembles,
                replica_exchange_mode: self.replica_exchange_mode,
                extra,
                chunk_size: self.chunk_size
            };
            Ok(
                rees
            )
            
        }
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
where Hist: HistogramVal<Energy> + HistogramCombine + Send + Sync,
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
where Hist: HistogramVal<Energy> + HistogramCombine + Send + Sync,
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

/// # Results of the simulation
/// This is what we do the simulation for!
/// 
/// * similar to [merged_log10_prob_and_aligned](`crate::rewl::Rewl::merged_log10_prob_and_aligned`)
/// * the difference is, that the logarithms are now calculated to base 10
pub fn merged_log10_probability_and_align<Ensemble, R, Hist, Energy, S, Res>(
    rewls: &[Rewl<Ensemble, R, Hist, Energy, S, Res>]
) -> GluedResult<Hist>
where Hist: HistogramCombine + HistogramVal<Energy> + Send + Sync,
    Energy: PartialOrd
{
    merged_log10_probability_and_align_ignore(rewls, &[])
}

/// # Results of the simulation
/// This is what we do the simulations for!
/// 
/// * similar to [merged_log10_probability_and_align](`crate::rewl::merged_log10_probability_and_align`)
/// * Now, however, we have a slice called `ignore`. It should contain the indices 
/// of all walkers, that should be ignored for the alignment and merging into the 
/// final probability density function. The indices do not need to be sorted, though
/// duplicates will be ignored and indices, which are out of bounds will also be ignored
pub fn merged_log10_probability_and_align_ignore<Ensemble, R, Hist, Energy, S, Res>(
    rewls: &[Rewl<Ensemble, R, Hist, Energy, S, Res>],
    ignore: &[usize]
) -> GluedResult<Hist>
where Hist: HistogramCombine + HistogramVal<Energy> + Send + Sync,
    Energy: PartialOrd
{
    let mut res = merged_log_probability_and_align_ignore(rewls, ignore)?;
    ln_to_log10(&mut res.1);
    res.2.par_iter_mut()
        .for_each(|slice| ln_to_log10(slice));
    Ok(res)
}

/// # Results of the simulation
/// This is what we do the simulations for!
/// 
/// * similar to [log_probability_and_align](`crate::rewl::log_probability_and_align`)
/// * Here, however, all logarithms are base 10
pub fn log10_probability_and_align<Ensemble, R, Hist, Energy, S, Res>(
    rewls: &[Rewl<Ensemble, R, Hist, Energy, S, Res>]
) -> GluedResult<Hist>
where Hist: HistogramCombine + HistogramVal<Energy> + Send + Sync,
    Energy: PartialOrd
{
    log10_probability_and_align_ignore(rewls, &[])
}

/// # Results of the simulation
/// This is what we do the simulations for!
/// 
/// * similar to [log10_probability_and_align](`crate::rewl::log10_probability_and_align`)
/// * Now, however, we have a slice called `ignore`. It should contain the indices 
/// of all walkers, that should be ignored for the alignment and merging into the 
/// final probability density function. The indices do not need to be sorted, though
/// duplicates will be ignored and indices, which are out of bounds will also be ignored
pub fn log10_probability_and_align_ignore<Ensemble, R, Hist, Energy, S, Res>(
    rewls: &[Rewl<Ensemble, R, Hist, Energy, S, Res>], ignore: &[usize]
) -> GluedResult<Hist>
where Hist: HistogramCombine + HistogramVal<Energy> + Send + Sync,
    Energy: PartialOrd
{
    let mut res = log_probability_and_align_ignore(rewls, ignore)?;
    ln_to_log10(&mut res.1);
    res.2.par_iter_mut()
        .for_each(|slice| ln_to_log10(slice));
    Ok(res)
}

/// # Results of the simulation
/// This is what we do the simulations for!
/// 
/// * `rewls` a slice of all replica exchange simulations you which to merge 
/// to create a final probability density estimate for whatever you sampled. 
/// Note, that while the slice `rewls` does not need to be ordered,
/// there should not be no gaps between the intervals that were sampled.
/// Also, the overlap of adjacent intervals should be large enough. 
/// 
/// # Result::Ok
/// * The Hist is only useful for the interval, i.e., it tells you which bins 
/// correspond to the entries of the probability density function - it does not count how often the bins were hit.
/// It is still the encapsulating interval, for which the probability density function was calculated
/// * The `Vec<f64>` is the logarithm (base e) of the probability density function, 
/// which you wanted to get!
///  * `Vec<Vec<f64>> these are the aligned probability estimates (also logarithm base e)
/// of the different intervals. 
/// This can be used to see, how good the simulation worked, e.g., by plotting them to see, if they match
/// 
/// # Failures
/// Failes if the internal histograms (intervals) do not align. 
/// Might fail if there is no overlap between neighboring intervals
/// 
/// # Notes
/// The difference between this function and 
/// [log_probability_and_align](`crate::rewl::log_probability_and_align`) is,
/// that, if there are multiple walkers in the same interval, they **will** be merged by 
/// averaging their probability estimates in this function, while they are **not** averaged in 
/// [log_probability_and_align](`crate::rewl::log_probability_and_align`)
pub fn merged_log_probability_and_align<Ensemble, R, Hist, Energy, S, Res>
(
    rewls: &[Rewl<Ensemble, R, Hist, Energy, S, Res>]
) -> GluedResult<Hist>
where Hist: HistogramCombine + HistogramVal<Energy> + Send + Sync,
    Energy: PartialOrd
{
    merged_log_probability_and_align_ignore(rewls, &[])
}

/// # Result of the simulation
/// This is what you were looking for!
/// 
/// * similar to [merged_log_probability_and_align](`crate::rewl::merged_log_probability_and_align`)
/// * Now, however, we have a slice called `ignore`. It should contain the indices 
/// of all walkers, that should be ignored for the alignment and merging into the 
/// final probability density function. The indices do not need to be sorted, though
/// duplicates will be ignored and indices, which are out of bounds will also be ignored
pub fn merged_log_probability_and_align_ignore<Ensemble, R, Hist, Energy, S, Res>(
    rewls: &[Rewl<Ensemble, R, Hist, Energy, S, Res>],
    ignore: &[usize]
) -> GluedResult<Hist>
where Hist: HistogramCombine + HistogramVal<Energy> + Send + Sync,
    Energy: PartialOrd
{
    if rewls.is_empty(){
        return Err(HistErrors::EmptySlice);
    }
    let merged_prob = merged_probs(rewls);
    let mut container = combine_container(rewls, &merged_prob, true);
    ignore_fn(&mut container, ignore);
    let (merge_points, alignment, log_prob, e_hist) = align(&container)?;
    merged_and_aligned(
        container.iter()
            .map(|c| c.1),
        merge_points,
        alignment,
        log_prob,
        e_hist
    )
}

/// # Results of the simulation
/// This is what we do the simulations for!
/// 
/// * `rewls` a slice of all replica exchange simulations you which to merge 
/// to create a final probability density estimate for whatever you sampled. 
/// Note, that while the slice `rewls` does not need to be ordered,
/// there should not be no gaps between the intervals that were sampled.
/// Also, the overlap of adjacent intervals should be large enough. 
/// 
/// # Result::Ok
/// * The Hist is only useful for the interval, i.e., it tells you which bins 
/// correspond to the entries of the probability density function - it does not count how often the bins were hit.
/// It is still the encapsulating interval, for which the probability density function was calculated
/// * The `Vec<f64>` is the logarithm (base e) of the probability density function, 
/// which you wanted to get!
///  * `Vec<Vec<f64>> these are the aligned probability estimates (also logarithm base e)
/// of the different intervals. 
/// This can be used to see, how good the simulation worked, e.g., by plotting them to see, if they match
/// 
/// # Failures
/// Failes if the internal histograms (intervals) do not align. 
/// Might fail if there is no overlap between neighboring intervals
/// 
/// # Notes
/// The difference between this function and 
/// [merged_log_probability_and_align](`crate::rewl::merged_log_probability_and_align`) is,
/// that, if there are multiple walkers in the same interval, they will **not** be merged by 
/// averaging their probability estimates in this function, while they **are averaged** in 
/// [merged_log_probability_and_align](`crate::rewl::merged_log_probability_and_align`)
pub fn log_probability_and_align<Ensemble, R, Hist, Energy, S, Res>(
    rewls: &[Rewl<Ensemble, R, Hist, Energy, S, Res>]
) -> GluedResult<Hist>
where Hist: HistogramCombine + HistogramVal<Energy> + Send + Sync,
    Energy: PartialOrd
{
    log_probability_and_align_ignore(rewls, &[])
}

/// # Results of the simulation
/// This is what we do the simulations for!
/// 
/// * similar to [log_probability_and_align](`crate::rewl::log_probability_and_align`)
/// * Now, however, we have a slice called `ignore`. It should contain the indices 
/// of all walkers, that should be ignored for the alignment and merging into the 
/// final probability density function. The indices do not need to be sorted, though
/// duplicates will be ignored and indices, which are out of bounds will also be ignored
pub fn log_probability_and_align_ignore<Ensemble, R, Hist, Energy, S, Res>(
    rewls: &[Rewl<Ensemble, R, Hist, Energy, S, Res>], ignore: &[usize]
) -> GluedResult<Hist>
where Hist: HistogramCombine + HistogramVal<Energy> + Send + Sync,
    Energy: PartialOrd
{
    if rewls.is_empty(){
        return Err(HistErrors::EmptySlice);
    }
    let probs = probs(rewls);
    let mut container = combine_container(rewls, &probs, false);
    ignore_fn(&mut container, ignore);

    let (merge_points, alignment, log_prob, e_hist) = align(&container)?;
    merged_and_aligned(
        container.iter()
            .map(|c| c.1),
        merge_points,
        alignment,
        log_prob,
        e_hist
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

fn probs<Ensemble, R, Hist, Energy, S, Res>
(
    rewls: &[Rewl<Ensemble, R, Hist, Energy, S, Res>]
) -> Vec<Vec<f64>>
{
    rewls.iter()
        .flat_map(
            |rewl| 
            {
                rewl.walkers()
                    .iter()
                    .map(
                        |w|
                            w.log_density().into()
                    )
           }
        ).collect()
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
