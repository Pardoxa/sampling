use std::sync::*;
use rand::{Rng, SeedableRng};
use std::num::NonZeroUsize;
use crate::*;

use rayon::prelude::*;

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};


fn merge_worker_prob<R, Hist>(worker: &mut [RewlWorker<R, Hist>])
{
    // The following if statement might be added later on - as of now it is unnessessary
    //if worker.len() <= 2 {
    //    return;
    //}
    let len = worker[0].probability_density.len();
    debug_assert![worker.iter().all(|w| w.probability_density.len() == len)];

    let num_workers_recip = (worker.len() as f64).recip();
    for i in 0..len {
        let mut val = worker[0].probability_density[i];
        for w in worker[1..].iter()
        {
            val += w.probability_density[i];
        }
        val *= num_workers_recip;
        for w in worker.iter_mut()
        {
            w.probability_density[i] = val;
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct RewlWorker<R, Hist>
{
    id: usize,
    sweep_size: NonZeroUsize,
    rng: R,
    hist: Hist,
    probability_density: Vec<f64>,
}

impl<R, Hist> RewlWorker<R, Hist> 
where R: Rng + Send + Sync,
    Self: Send + Sync,
    Hist: Histogram
{
    pub fn new(id: usize, rng: R, hist: Hist, sweep_size: NonZeroUsize) -> RewlWorker<R, Hist> {
        let probability_density = vec![0.0; hist.bin_count()];
        RewlWorker{
            id,
            rng,
            hist,
            probability_density,
            sweep_size
        }
    }

    pub fn do_work<Ensemble>(&mut self, ensemble_vec: &[RwLock<Ensemble>])
    {
        let mut e = ensemble_vec[self.id]
            .write()
            .expect("Fatal Error encountered; ERRORCODE 0x1 - this should be \
                impossible to reach. If you are using the latest version of the \
                'sampling' library, please contact the library author via github by opening an \
                issue! https://github.com/Pardoxa/sampling/issues");
        for _ in 0..self.sweep_size.get()
        {

        }
    }
}


#[derive(Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Rewl<Ensemble, R, Hist>{
    chunk_size: NonZeroUsize,
    ensembles: Vec<RwLock<Ensemble>>,
    map: Vec<usize>,
    worker: Vec<RewlWorker<R, Hist>>
}

impl<Ensemble, R, Hist> Rewl<Ensemble, R, Hist> 
where R: Send + Sync + Rng,
    Hist: Send + Sync + Histogram
{

    pub fn new_from_other_rng<R2>(mut rng: &mut R2, ensembles: Vec<Ensemble>, hists: Vec<Hist>, chunk_size: NonZeroUsize, sweep_size: NonZeroUsize) -> Rewl<Ensemble, R, Hist>
    where R2: Rng,
        R: SeedableRng,
    {
        let map = (0..ensembles.len()).collect();
        
        let worker = (0..ensembles.len())
            .zip(hists.into_iter())
            .map(|(id, hist)| 
                {
                    let r = R::from_rng(&mut rng)
                        .expect("Unable to seed Rng");
                    RewlWorker::new(id, r, hist, sweep_size)
                }
            )
            .collect();
        
        let e = ensembles.into_iter()
            .map(|val| RwLock::new(val))
            .collect();
        Rewl{
            map,
            worker,
            ensembles: e,
            chunk_size
        }
    }

    pub fn new(rng: &mut R, ensembles: Vec<Ensemble>, hists: Vec<Hist>, chunk_size: NonZeroUsize, sweep_size: NonZeroUsize) -> Rewl<Ensemble, R, Hist>
    where R: Rng + SeedableRng
    {
       Self::new_from_other_rng(rng, ensembles, hists, chunk_size, sweep_size)
    }

    pub fn sweep_test(&mut self)
    where Ensemble: TestEnsemble + Send + Sync,
        R: Send + Sync
    {
        let slice = self.ensembles.as_slice();
        self.worker
            .par_iter_mut()
            .for_each(|w| w.do_work(slice));


        if self.chunk_size.get() >= 2 {
            self.worker
                .par_chunks_mut(self.chunk_size.get())
                .for_each(merge_worker_prob);
        }
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

#[cfg(test)]
mod tests
{
    use super::*;
    use rand_pcg::Pcg64Mcg;


    #[test]
    fn proof_of_concept()
    {
        let ensembles = (0..50)
            .map(|num| TestEns::new(num))
            .collect();
        
        let mut rng = Pcg64Mcg::seed_from_u64(94375982592);

        let mut rewl = Rewl::new(&mut rng, ensembles, 1);

        for _ in 0..10 {
            rewl.sweep_test();
            println!()
        }
    }
}