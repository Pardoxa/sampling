use crate::*;
use crate::rewl::*;
use crate::glue_helper::*;
use rand::{Rng, SeedableRng, prelude::SliceRandom};
use std::{num::NonZeroUsize, sync::*};
use rayon::prelude::*;

#[cfg(feature = "rewl_sweep_time_optimization")]
use std::cmp::Reverse;

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

/// # Efficient replica exchange Wang landau
/// * use this to quickly build your own parallel replica exchange wang landau simulation
/// ## Tipp
/// Use the short hand `Rewl`  
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

    /// # Get the number of intervals present
    pub fn num_intervals(&self) -> usize
    {
        self.walker.len() / self.chunk_size.get()
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

    /// # Change sweep size for markov chain of walkers
    /// * changes the sweep size used in the sweep
    /// * changes sweep size of all walkers in the nth interval
    /// * returns Err if index out of bounds, i.e., the requested interval does not exist
    /// * interval counting starts at 0, i.e., n=0 is the first interval
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
        condition: C
    )
    where 
        Ensemble: Send + Sync,
        R: Send + Sync,
        F: Fn(&mut Ensemble) -> Option<Energy> + Copy + Send + Sync,
        C: Fn(&Self) -> bool
    {
        while !self.is_finished() && condition(&self)
        {
            self.sweep(energy_fn);
        }
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

        #[cfg(not(feature = "rewl_sweep_time_optimization"))]
        let walker = &mut self.walker;

        #[cfg(feature = "rewl_sweep_time_optimization")]
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
        log_prob.iter_mut()
            .for_each(|val| *val *= std::f64::consts::LOG10_E);

        Ok((e_hist, log_prob))

    }

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
    pub fn merged_log10_prob_and_aligned(&self) -> Result<(Hist, Vec<f64>, Vec<Vec<f64>>), HistErrors>
    where Hist: HistogramCombine
    {
        let (e_hist, mut log_prob, mut aligned) = self.merged_log_prob_and_aligned()?;
        log_prob.iter_mut()
            .for_each(|val| *val *= std::f64::consts::LOG10_E);
        
        aligned.par_iter_mut()
            .for_each(
                |a| 
                {
                    a.iter_mut()
                        .for_each(|val| *val *= std::f64::consts::LOG10_E)
                }
            );
        Ok(
            (e_hist, log_prob, aligned)
        )
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
    pub fn merged_log_prob_and_aligned(&self) -> Result<(Hist, Vec<f64>, Vec<Vec<f64>>), HistErrors>
    where Hist: HistogramCombine 
    {
        let (e_hist, mut log_prob, mut aligned) = self.merged_log_probability_and_align()?;
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
        aligned.par_iter_mut()
            .for_each(
                |aligned|
                {
                    aligned.iter_mut()
                        .for_each(|val| *val -= sum)
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
        let (merge_points, alignment, log_prob, e_hist) = self.merged_log_probability_helper()?;
        Ok(
            Self::only_merged(
                merge_points,
                alignment,
                log_prob,
                e_hist
            )
        )
    }

    fn merged_log_probability_and_align(&self) -> Result<(Hist, Vec<f64>, Vec<Vec<f64>>), HistErrors>
    where Hist: HistogramCombine
    {
        let (merge_points, alignment, log_prob, e_hist) = self.merged_log_probability_helper()?;
        self.merged_and_aligned(
            merge_points,
            alignment,
            log_prob,
            e_hist
        )
    }

    fn merged_log_probability_helper(&self) -> Result<(Vec<usize>, Vec<usize>, Vec<Vec<f64>>, Hist), HistErrors>
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

        Ok(
            (
                merge_points,
                alignment,
                log_prob,
                e_hist
            )
        )

    }

    fn only_merged(
        merge_points: Vec<usize>,
        alignment: Vec<usize>,
        log_prob: Vec<Vec<f64>>,
        e_hist: Hist
    ) -> (Vec<f64>, Hist)
    {
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

        (merged_log_prob, e_hist)
    }

    fn merged_and_aligned(
        &self,
        merge_points: Vec<usize>,
        alignment: Vec<usize>,
        log_prob: Vec<Vec<f64>>,
        e_hist: Hist
    ) -> Result<(Hist, Vec<f64>, Vec<Vec<f64>>), HistErrors>
    where Hist: HistogramCombine
    {
        let mut merged_log_prob = vec![f64::NAN; e_hist.bin_count()];

        let mut aligned_intervals = vec![merged_log_prob.clone(); alignment.len() + 1];

        aligned_intervals[0][..log_prob[0].len()]
            .copy_from_slice(&log_prob[0]);
        
        merged_log_prob[..=merge_points[0]]
            .copy_from_slice(&log_prob[0][..=merge_points[0]]);


        // https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=2dcb7b7a3be78397d34657ece42aa851
        let mut align_sum = 0;

        for ((index, (&a, &mp)), hist) in alignment.iter()
            .zip(merge_points.iter()).enumerate()
            .zip(
                self.walker.iter()
                    .step_by(self.walkers_per_interval().get())
                    .skip(1)
                    .map(|v| v.hist())
            )
        {
            let position_l = mp + align_sum;
            align_sum += a;
            let left = mp - a;

            let index_p1 = index + 1; 

            let shift = merged_log_prob[position_l] - log_prob[index_p1][left];

            let unmerged_align = e_hist.align(hist)?;

            aligned_intervals[index_p1][unmerged_align..]
                .iter_mut()
                .zip(log_prob[index_p1].iter())
                .for_each(|(v, &val)| *v = val + shift);

            merged_log_prob[position_l..]
                .iter_mut()
                .zip(log_prob[index_p1][left..].iter())
                .for_each(
                    |(merge, val)|
                    {
                        *merge = val + shift;
                    }
                );


        }
        Ok(
            (e_hist, merged_log_prob, aligned_intervals)
        )
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

    /// # read access to your ensembles
    /// * If you do not know what `RwLockReadGuard<'a, Ensemble>` is - do not worry.
    /// you can just pretend it is `&Ensemble` and everything will work out fine
    pub fn ensembles<'a>(&'a self) -> Vec<RwLockReadGuard<'a, Ensemble>>
    {
        self.ensembles.iter()
            .map(|e| e.read().unwrap())
            .collect()
    }

    /// # read access to the internal histograms used by the walkers
    pub fn hists(&self) -> Vec<&Hist>
    {
        self.walker.iter()
            .map(|w| w.hist())
            .collect()
    }

    /// # Read access to internal rewl walkers
    /// * each of these walkers independently samples an interval. 
    /// * see paper for more infos
    pub fn walker(&self) -> &Vec<RewlWalker<R, Hist, Energy, S, Res>>
    {
        &self.walker
    }
}
