
use std::{borrow::Borrow, num::NonZeroUsize};
use crate::histogram::*;
use super::{
    glue::*,
    glue_helper::{
        ln_to_log10,
        log10_to_ln
    },
    LogBase
};

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

pub trait GlueAble<H>{
    fn push_glue_entry(&self, job: &mut GlueJob<H>)
    {
        self.push_glue_entry_ignoring(job, &[])
    }

    fn push_glue_entry_ignoring(
        &self, 
        job: &mut GlueJob<H>,
        ignore_idx: &[usize]
    );
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum SimulationType{
    WangLandau1T = 0,
    WangLandau1TAdaptive = 1,
    Entropic = 2,
    EntropicAdaptive = 3,
    REWL = 4,
    REES = 5,
    Unknown = 6
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum SimProgress{
    LogF(f64),
    MissingSteps(u64),
    Unknown
}

/// Statistics of one interval, used to gauge how well
/// the simulation works etc.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct IntervalSimStats{
    /// the progress of the Interval
    pub sim_progress: SimProgress,
    /// Which type of simulation did the interval come from
    pub interval_sim_type: SimulationType,
    /// How many steps were rejected in total in the interval
    pub rejected_steps: u64,
    /// How many steps were accepted in total in the interval
    pub accepted_steps: u64,
    /// How many replica exchanges were performed?
    /// None for Simulations that don't do replica exchanges
    pub replica_exchanges: Option<u64>,
    /// How many replica exchanges were proposed?
    /// None for simulations that do not perform replica exchanges
    pub proposed_replica_exchanges: Option<u64>,
    /// The number of walkers used to generate this sim.
    /// In Replica exchange sims you can have more than one walker 
    /// per interval, which is where this comes from
    pub merged_over_walkers: NonZeroUsize
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct GlueEntry<H>{
    pub hist: H,
    pub prob: Vec<f64>,
    pub log_base: LogBase,
    pub interval_stats: IntervalSimStats
}

impl<H> Borrow<H> for GlueEntry<H>
{
    fn borrow(&self) -> &H {
        &self.hist
    }
}

/// # Used to merge probability densities from WL, REWL, Entropic or REES simulations
/// * You can also mix those methods and still glue them
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct GlueJob<H>
{
    pub collection: Vec<GlueEntry<H>>,
    pub round_trips: Vec<usize>,
    pub desired_logbase: LogBase
}

impl<H> GlueJob<H>
    where H: Clone
{
    pub fn new<B>(
        to_glue: &B,
        desired_logbase: LogBase
    ) -> Self
    where B: GlueAble<H>
    {
        let mut job = Self { 
            collection: Vec::new(),
            round_trips: Vec::new(),
            desired_logbase
        };

        to_glue.push_glue_entry(&mut job);
        job
    }

    pub fn new_from_slice<B>(to_glue: &[B], desired_logbase: LogBase) -> Self
        where B: GlueAble<H>
    {
        Self::new_from_iter(to_glue.iter(), desired_logbase)
    }

    pub fn new_from_iter<'a, B, I>(
        to_glue: I,
        desired_logbase: LogBase
    ) -> Self
    where B: GlueAble<H> + 'a,
    I: Iterator<Item=&'a B> 
    {
        let mut job = Self { 
            collection: Vec::new(),
            round_trips: Vec::new(),
            desired_logbase
        };

        job.add_iter(to_glue);
        job
    }

    pub fn add_slice<B>(&mut self, to_glue: &[B])
        where B: GlueAble<H>
    {
        self.add_iter(to_glue.iter())
    }

    pub fn add_iter<'a, I, B>(&mut self, to_glue: I)
    where B: GlueAble<H> + 'a,
        I: Iterator<Item=&'a B> 
    {
        for entry in to_glue {
            entry.push_glue_entry(self);
        }
    }

    pub fn get_stats(&self) -> GlueStats
    {
        let interval_stats = self
            .collection
            .iter()
            .map(|e| e.interval_stats.clone())
            .collect();
        GlueStats{
            interval_stats,
            roundtrips: self.round_trips.clone()
        }
    }

    /// # Calculate the probability density function from overlapping intervals
    /// 
    /// This uses a average merge, which first align all intervals and then merges 
    /// the probability densities by averaging in the logarithmic space
    /// 
    /// The [Glued] allows you to easily write the probability density function to a file
    pub fn average_merged_and_aligned<T>(&mut self) -> Result<Glued<H, T>, HistErrors>
    where H: Histogram + HistogramCombine + HistogramVal<T>,
        T: PartialOrd{

        let log_prob = self.prepare_for_merge()?;
        let mut res = average_merged_and_aligned(
            log_prob, 
            &self.collection, 
            self.desired_logbase
        )?;
        let stats = self.get_stats();
        res.set_stats(stats);
        Ok(res)
    }

    /// # Calculate the probability density function from overlapping intervals
    /// 
    /// This uses a derivative merge
    /// 
    /// The [Glued] allows you to easily write the probability density function to a file
    pub fn derivative_glue_and_align<T>(&mut self) -> Result<Glued<H, T>, HistErrors>
    where H: Histogram + HistogramCombine + HistogramVal<T>,
        T: PartialOrd{

        let log_prob = self.prepare_for_merge()?;
        let mut res = derivative_merged_and_aligned(
            log_prob, 
            &self.collection, 
            self.desired_logbase
        )?;
        let stats = self.get_stats();
        res.set_stats(stats);
        Ok(res)
    }

    fn prepare_for_merge<T>(
        &mut self
    ) -> Result<Vec<Vec<f64>>, HistErrors>
    where H: Histogram + HistogramCombine + HistogramVal<T>,
    T: PartialOrd
    {
        self.make_entries_desired_logbase();
        
        let mut encountered_invalid = false;

        self.collection
            .sort_unstable_by(
                |a, b|
                {
                    match a.hist
                        .first_border()
                        .partial_cmp(
                            &b.hist.first_border()
                        ){
                        None => {
                            encountered_invalid = true;
                            std::cmp::Ordering::Less
                        },
                        Some(o) => o
                    }
                }
            );
        if encountered_invalid {
            return Err(HistErrors::InvalidVal);
        }

        Ok(
            self.collection
                .iter()
                .map(|e| e.prob.clone())
                .collect()
        )

    }

    fn make_entries_desired_logbase(&mut self)
    {
        for e in self.collection.iter_mut()
        {
            match self.desired_logbase{
                LogBase::Base10 => {
                    if e.log_base.is_base_e(){
                        e.log_base = LogBase::Base10;
                        ln_to_log10(&mut e.prob)
                    }
                },
                LogBase::BaseE => {
                    if e.log_base.is_base10() {
                        e.log_base = LogBase::BaseE;
                        log10_to_ln(&mut e.prob)
                    }
                }
            }
        }
    }
}