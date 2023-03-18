
use std::borrow::Borrow;

use crate::histogram::*;

use super::{
    replica_glued::*,
    glue_helper::{
        ln_to_log10,
        log10_to_ln
    },
    LogBase
};

pub trait GlueAble<H>{
    fn glue_entry(&self) -> GlueEntry::<H>;
}

pub struct GlueEntry<H>{
    pub(crate) hist: H,
    pub(crate) prob: Vec<f64>,
    pub(crate) log_base: LogBase
}

impl<H> Borrow<H> for GlueEntry<H>
{
    fn borrow(&self) -> &H {
        &self.hist
    }
}

/// # Used to merge probability densities from WL, REWL, Entropic or REES simulations
/// * You can also mix those methods and still glue them
pub struct GlueJob<H>
{
    collection: Vec<GlueEntry<H>>,
    desired_logbase: LogBase
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
        Self { 
            collection: vec![to_glue.glue_entry()],
            desired_logbase
        }
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
        let collection = to_glue
            .map(GlueAble::glue_entry)
            .collect();
        Self{
            collection,
            desired_logbase
        }
    }

    pub fn add<B>(&mut self, to_glue: &B)
    where B: GlueAble<H>
    {
        self.collection
            .push(
                to_glue.glue_entry()
            )
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
        self.collection
            .extend(
                to_glue.map(|e| e.glue_entry())
            );
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
        average_merged_and_aligned(
            log_prob, 
            &self.collection, 
            self.desired_logbase
        )
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
        derivative_merged_and_aligned(
            log_prob, 
            &self.collection, 
            self.desired_logbase
        )
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



#[cfg(test)]
mod tests{
    use super::*;
    pub struct TestGlueable {
        pub hist: HistI64Fast,
        pub prob: Vec<f64>
    }
    
    impl GlueAble<HistI64Fast> for TestGlueable
    {
        fn glue_entry(&self) -> GlueEntry::<HistI64Fast> {
            GlueEntry { 
                hist: self.hist.clone(), 
                prob: self.prob.clone(), 
                log_base: LogBase::Base10
            }
        }
    }

    #[test]
    fn glue_test()
    {
        let gl = TestGlueable{
            hist: HistI64Fast::new_inclusive(1, 10).unwrap(),
            prob: vec![0.0;10]
        };
    }
}