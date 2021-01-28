use rand::Rng;
use std::{num::NonZeroUsize, marker::PhantomData, sync::*, mem::*};
use crate::*;
use crate::wang_landau::WangLandauMode;

use rayon::prelude::*;

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};


fn merge_walker_prob<R, Hist, Energy, S, Res>(walker: &mut [RewlWalker<R, Hist, Energy, S, Res>])
{
    // The following if statement might be added later on - as of now it is unnessessary
    //if walker.len() <= 2 {
    //    return;
    //}
    let len = walker[0].log_density.len();
    debug_assert![walker.iter().all(|w| w.log_density.len() == len)];

    let num_walkers_recip = (walker.len() as f64).recip();
    for i in 0..len {
        let mut val = walker[0].log_density[i];
        for w in walker[1..].iter()
        {
            val += w.log_density[i];
        }
        val *= num_walkers_recip;
        for w in walker.iter_mut()
        {
            w.log_density[i] = val;
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct RewlWalker<R, Hist, Energy, S, Res>
{
    id: usize,
    sweep_size: NonZeroUsize,
    rng: R,
    hist: Hist,
    log_density: Vec<f64>,
    log_f: f64,
    step_count: usize,
    mode: WangLandauMode,
    old_energy: Energy,
    bin: usize,
    marker_s: PhantomData<S>,
    marker_res: PhantomData<Res>,
}

fn replica_exchange<R, Hist, Energy, S, Res>
(
    walker_a: &mut RewlWalker<R, Hist, Energy, S, Res>,
    walker_b: &mut RewlWalker<R, Hist, Energy, S, Res>
) where Hist: HistogramVal<Energy>,
    R: Rng
{
    // check if exchange is even possible
    let new_bin_a = match walker_a.hist.get_bin_index(&walker_b.old_energy)
    {
        Ok(bin) => bin,
        _ => return,
    };

    let new_bin_b = match walker_b.hist.get_bin_index(&walker_a.old_energy)
    {
        Ok(bin) => bin,
        _ => return,
    };

    // see paper equation 1
    let log_gi_x = walker_a.log_density[walker_a.bin];
    let log_gi_y = walker_a.log_density[new_bin_a];

    let log_gj_y = walker_b.log_density[walker_b.bin];
    let log_gj_x = walker_b.log_density[new_bin_b];

    let log_prob = log_gi_x + log_gj_y - log_gi_y - log_gj_x;

    let prob = log_prob.exp();

    // if exchange is accepted
    if walker_b.rng.gen::<f64>() < prob 
    {
        swap(&mut walker_b.id, &mut walker_a.id);
        swap(&mut walker_b.old_energy, &mut walker_a.old_energy);
        walker_b.bin = new_bin_b;
        walker_a.bin = new_bin_a;
    }
}

impl<R, Hist, Energy, S, Res> RewlWalker<R, Hist, Energy, S, Res> 
where R: Rng + Send + Sync,
    Self: Send + Sync,
    Hist: Histogram + HistogramVal<Energy>,
    Energy: Send + Sync,
    S: Send + Sync,
    Res: Send + Sync
{
    pub fn new
    (
        id: usize,
        rng: R,
        hist: Hist,
        sweep_size: NonZeroUsize,
        old_energy: Energy
    ) -> RewlWalker<R, Hist, Energy, S, Res>
    {
        let log_density = vec![0.0; hist.bin_count()];
        let bin = hist.get_bin_index(&old_energy).unwrap();
        RewlWalker{
            id,
            rng,
            hist,
            log_density,
            sweep_size,
            log_f: 1.0,
            step_count: 0,
            mode: WangLandauMode::RefineOriginal,
            old_energy,
            bin,
            marker_res: PhantomData::<Res>,
            marker_s: PhantomData::<S>
        }
    }

    pub fn log_f(&self) -> f64
    {
        self.log_f
    }

    fn log_f_1_t(&self) -> f64
    {
        self.hist.bin_count() as f64 / self.step_count as f64
    }

    pub fn wang_landau_sweep<Ensemble, F>
    (
        &mut self,
        ensemble_vec: &[RwLock<Ensemble>],
        step_size: usize,
        energy_fn: F
    )
    where F: Fn(&mut Ensemble) -> Energy,
        Ensemble: MarkovChain<S, Res>
    {
        let mut e = ensemble_vec[self.id]
            .write()
            .expect("Fatal Error encountered; ERRORCODE 0x1 - this should be \
                impossible to reach. If you are using the latest version of the \
                'sampling' library, please contact the library author via github by opening an \
                issue! https://github.com/Pardoxa/sampling/issues");
        
        let mut steps = Vec::with_capacity(step_size);
        for _ in 0..self.sweep_size.get()
        {   
            e.m_steps(step_size, &mut steps);
            let energy = energy_fn(&mut e);

            self.step_count = self.step_count.saturating_add(1);
            
            if self.mode.is_mode_1_t() {
                self.log_f = self.log_f_1_t();
            }

            match self.hist.get_bin_index(&energy) 
            {
                Ok(current_bin) => {
                    // metropolis hastings
                    let acception_prob = (self.log_density[self.bin] - self.log_density[current_bin])
                        .exp();
                    if self.rng.gen::<f64>() > acception_prob 
                    {
                        e.undo_steps_quiet(&steps);
                    } else {
                        self.old_energy = energy;
                        self.bin = current_bin;
                    }
                },
                _ => {
                    e.undo_steps_quiet(&steps);
                }
            }

            self.hist.count_index(self.bin)
                .expect("Histogram index Error, ERRORCODE 0x2");
            
            self.log_density[self.bin] += self.log_f;
        }

        // Check if log_f should be halfed or mode should be changed
        if self.mode.is_mode_original() && !self.hist.any_bin_zero() {
            let ref_1_t = self.log_f_1_t();
            self.log_f *= 0.5;

            if self.log_f < ref_1_t {
                self.log_f = ref_1_t;
                self.mode = WangLandauMode::Refine1T;
            }
            self.hist.reset();
        }
    }
}


#[derive(Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Rewl<Ensemble, R, Hist, Energy, S, Res>{
    chunk_size: NonZeroUsize,
    ensembles: Vec<RwLock<Ensemble>>,
    map: Vec<usize>,
    walker: Vec<RewlWalker<R, Hist, Energy, S, Res>>,
    log_f_threshold: f64,
    replica_exchange_mode: bool,
}

impl<Ensemble, R, Hist, Energy, S, Res> Rewl<Ensemble, R, Hist, Energy, S, Res> 
where R: Send + Sync + Rng,
    Hist: Send + Sync + Histogram + HistogramVal<Energy>,
    Energy: Send + Sync,
    Ensemble: MarkovChain<S, Res>,
    Res: Send + Sync,
    S: Send + Sync
{

    //pub fn new_from_other_rng<R2>
    //(
    //    mut rng: &mut R2, 
    //    ensembles: Vec<Ensemble>, 
    //    hists: Vec<Hist>, 
    //    chunk_size: NonZeroUsize, 
    //    sweep_size: NonZeroUsize
    //) -> Rewl<Ensemble, R, Hist, Energy>
    //where R2: Rng,
    //    R: SeedableRng,
    //    Hist: HistogramVal<Energy>
    //{
    //    let map = (0..ensembles.len()).collect();
    //    
    //    let walker = (0..ensembles.len())
    //        .zip(hists.into_iter())
    //        .map(|(id, hist)| 
    //            {
    //                let r = R::from_rng(&mut rng)
    //                    .expect("Unable to seed Rng");
    //                RewlWalker::new(id, r, hist, sweep_size)
    //            }
    //        )
    //        .collect();
    //    
    //    let e = ensembles.into_iter()
    //        .map(|val| RwLock::new(val))
    //        .collect();
    //    Rewl{
    //        map,
    //        walker,
    //        ensembles: e,
    //        chunk_size
    //    }
    //}
//
    //pub fn new(rng: &mut R, ensembles: Vec<Ensemble>, hists: Vec<Hist>, chunk_size: NonZeroUsize, sweep_size: NonZeroUsize) -> Rewl<Ensemble, R, Hist>
    //where R: Rng + SeedableRng,
    //    Hist: HistogramVal<Energy>
    //{
    //   Self::new_from_other_rng(rng, ensembles, hists, chunk_size, sweep_size)
    //}

    pub fn sweep<F>(&mut self, step_size: usize, energy_fn: F)
    where Ensemble: TestEnsemble + Send + Sync,
        R: Send + Sync,
        F: Fn(&mut Ensemble) -> Energy + Copy + Send + Sync
    {
        let slice = self.ensembles.as_slice();
        self.walker
            .par_iter_mut()
            .for_each(|w| w.wang_landau_sweep(slice, step_size, energy_fn));


        if self.chunk_size.get() >= 2 {
            self.walker
                .par_chunks_mut(self.chunk_size.get())
                .for_each(merge_walker_prob);
        }

        // replica exchange
        let walker_slice = if self.replica_exchange_mode 
        {
            &mut self.walker
        } else {
            &mut self.walker[self.chunk_size.get()..]
        };

        self.replica_exchange_mode = !self.replica_exchange_mode;
        let chunk_size = self.chunk_size.get();
        walker_slice
            .par_chunks_exact_mut(2 * self.chunk_size.get())
            .for_each(
                |walker_chunk|
                {
                    let (slice_a, slice_b) = walker_chunk.split_at_mut(chunk_size);
                    for (walker_a, walker_b) in slice_a.iter_mut()
                        .zip(slice_b.iter_mut())
                    {
                        replica_exchange(walker_a, walker_b);
                    }
                }
            )
        
        
    }
}


pub trait TestEnsemble{
    fn num(&self) -> usize;
    fn add_one(&mut self);
}

struct TestEns{
    num: usize
}

impl TestEns{
    #[allow(dead_code)]
    pub fn new(id: usize) -> Self {
        TestEns{
            num: id
        }
    }
}

impl TestEnsemble for TestEns
{
    fn num(&self) -> usize {
        self.num
    }

    fn add_one(&mut self) {
        self.num = self.num.wrapping_add(1);
    }
}

//#[cfg(test)]
//mod tests
//{
//    use super::*;
//    use rand_pcg::Pcg64Mcg;
//
//
//    #[test]
//    fn proof_of_concept()
//    {
//        let ensembles = (0..50)
//            .map(|num| TestEns::new(num))
//            .collect();
//        
//        let mut rng = Pcg64Mcg::seed_from_u64(94375982592);
//
//        let mut rewl = Rewl::new(&mut rng, ensembles, 1);
//
//        for _ in 0..10 {
//            rewl.sweep_test();
//            println!()
//        }
//    }
//}