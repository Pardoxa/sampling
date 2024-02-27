
use std::{borrow::Borrow, num::NonZeroUsize};
use crate::histogram::*;
use super::{
    glue_writer::*,
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

impl SimulationType
{
    /// # Name of simulation type as &str
    pub fn name(self) -> &'static str
    {
        match self{
            Self::Entropic => "Entropic",
            Self::WangLandau1T => "WangLandau1T",
            Self::EntropicAdaptive => "EntropicAdaptive",
            Self::WangLandau1TAdaptive => "WangLandau1TAdaptive",
            Self::REES => "REES",
            Self::REWL => "REWL",
            Self::Unknown => "Unknown"
        }
    }

    pub(crate) fn from_usize(num: usize) -> Self
    {
        match num{
            0 => Self::WangLandau1T,
            1 => Self::WangLandau1TAdaptive,
            2 => Self::Entropic,
            3 => Self::EntropicAdaptive,
            4 => Self::REWL,
            5 => Self::REES,
            6 => Self::Unknown,
            _ => unreachable!()
        }
    } 
}

pub(crate) struct AccumulatedIntervalStats{
    worst_log_progress: f64,
    worst_missing_steps_progress: u64,
    log_progress_counter: u32,
    missing_steps_progress_counter: u32,
    unknown_progress_counter: u32,
    interval_sim_type_counter: [usize; 7],
    total_rejected_steps: u64,
    total_accepted_steps: u64,
    total_proposed_replica_exchanges: u64,
    total_replica_exchanges: u64,
    potential_for_replica_exchanges: bool,
    potential_for_proposed_replica_exchanges: bool
}

impl AccumulatedIntervalStats{

    pub(crate) fn write<W: std::io::Write>(&self, mut writer: W) -> std::io::Result<()>
    {
        let total_intervals: usize = self
            .interval_sim_type_counter
            .iter()
            .sum();
        writeln!(writer, "#Accumulated Stats of {total_intervals} Intervals")?;
        if self.log_progress_counter > 0 {
            writeln!(
                writer,
                "#Worst log progress: {} - out of {} intervals that tracked log progress",
                self.worst_log_progress,
                self.log_progress_counter
            )?;
        }
        if self.missing_steps_progress_counter > 0 {
            writeln!(
                writer,
                "#Worst missing steps progress: {} - out of {} intervals that tracked missing steps progress",
                self.worst_missing_steps_progress,
                self.missing_steps_progress_counter
            )?;
        }
        if self.unknown_progress_counter > 0 {
            writeln!(writer, "# {} Intervals had unknown progress", self.unknown_progress_counter)?
        }
        
        for (index, &amount) in self.interval_sim_type_counter.iter().enumerate()
        {
            if amount > 0 {
                let sim_type = SimulationType::from_usize(index);
                writeln!(writer, "#{} contributed {} intervals", sim_type.name(), amount)?;
            }
        }

        let a = self.total_accepted_steps;
        let r = self.total_rejected_steps;
        let total = a + r;
        writeln!(writer, "#TOTAL: {a} accepted and {r} rejected steps, which makes a total of {total} steps")?;
        let a_rate = a as f64 / total as f64;
        writeln!(writer, "#TOTAL acceptance rate {a_rate}")?;
        let r_rate = r as f64 / total as f64;
        writeln!(writer, "#TOTAL rejection rate {r_rate}")?;

        if self.potential_for_replica_exchanges {
            writeln!(writer, "#TOTAL performed replica exchanges: {}", self.total_replica_exchanges)?;
        }
        if self.potential_for_proposed_replica_exchanges
        {
            writeln!(writer, "#TOTAL proposed replica exchanges: {}", self.total_proposed_replica_exchanges)?;
            if self.potential_for_replica_exchanges{
                let rate = self.total_replica_exchanges as f64 / self.total_proposed_replica_exchanges as f64;
                writeln!(writer, "#rate of accepting replica exchanges: {rate}")?;
            }
        }
        Ok(())
    }

    pub(crate) fn generate_stats(interval_stats: &[IntervalSimStats]) -> Self
    {
        let mut acc = AccumulatedIntervalStats{
            worst_log_progress: f64::NEG_INFINITY,
            worst_missing_steps_progress: 0,
            log_progress_counter: 0,
            missing_steps_progress_counter: 0,
            unknown_progress_counter: 0,
            interval_sim_type_counter: [0;7],
            total_accepted_steps: 0,
            total_rejected_steps: 0,
            total_proposed_replica_exchanges: 0,
            total_replica_exchanges: 0,
            potential_for_proposed_replica_exchanges: false,
            potential_for_replica_exchanges: false
        };

        for stats in interval_stats.iter()
        {
            acc.interval_sim_type_counter[stats.interval_sim_type as usize] += 1;
            match stats.sim_progress{
                SimProgress::LogF(log_f) => {
                    acc.log_progress_counter += 1;
                    acc.worst_log_progress = acc.worst_log_progress.max(log_f);
                },
                SimProgress::MissingSteps(missing) => {
                    acc.missing_steps_progress_counter += 1;
                    acc.worst_missing_steps_progress = acc.worst_missing_steps_progress.max(missing);
                },
                SimProgress::Unknown => {
                    acc.unknown_progress_counter += 1;
                }
            }

            acc.total_accepted_steps += stats.accepted_steps;
            acc.total_rejected_steps += stats.rejected_steps;
            if let Some(replica) = stats.replica_exchanges{
                acc.potential_for_replica_exchanges = true;
                acc.total_replica_exchanges += replica;
            }
            if let Some(proposed) = stats.proposed_replica_exchanges
            {
                acc.potential_for_proposed_replica_exchanges = true;
                acc.total_proposed_replica_exchanges += proposed;
            }
        }
        acc
    }
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

impl IntervalSimStats{
    pub fn write<W: std::io::Write>(&self, mut writer: W) -> std::io::Result<()>
    {
        writeln!(writer, "#Simulated via: {:?}", self.interval_sim_type.name())?;
        writeln!(writer, "#progress {:?}", self.sim_progress)?;
        if self.merged_over_walkers.get() == 1 {
            writeln!(writer, "#created from a single walker")?;
        } else {
            writeln!(writer, "#created from merging {} walkers", self.merged_over_walkers)?;
        }
        
        let a = self.accepted_steps;
        let r = self.rejected_steps;
        let total = a + r;
        writeln!(writer, "#had {a} accepted and {r} rejected steps, which makes a total of {total} steps")?;
        let a_rate = a as f64 / total as f64;
        writeln!(writer, "#acceptance rate {a_rate}")?;
        let r_rate = r as f64 / total as f64;
        writeln!(writer, "#rejection rate {r_rate}")?;

        if let Some(replica) = self.replica_exchanges {
            writeln!(writer, "#performed replica exchanges: {replica}")?;
        }
        if let Some(proposed) = self.proposed_replica_exchanges
        {
            writeln!(writer, "#proposed replica exchanges: {proposed}")?;
            if let Some(replica) = self.replica_exchanges{
                let rate = replica as f64 / proposed as f64;
                writeln!(writer, "#rate of accepting replica exchanges: {rate}")?;
            }
        }
        Ok(())
    }
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