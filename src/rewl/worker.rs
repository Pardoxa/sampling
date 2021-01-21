use std::sync::*;
use rand::{Rng, SeedableRng};

use rayon::prelude::*;

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

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

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct RewlWorker<R>
{
    id: usize,
    rng: R
}

impl<R> RewlWorker<R> 
where R: Rng + Send + Sync
{
    pub fn new(id: usize, rng: R) -> RewlWorker<R> {
        RewlWorker{
            id,
            rng
        }
    }

    pub fn do_work<Ensemble>(&mut self, ensemble_vec: &[RwLock<Ensemble>])
    where Ensemble: TestEnsemble
    {
        let mut e = ensemble_vec[self.id]
            .write()
            .expect("Fatal Error encountered; ERRORCODE 0x1 - this should be \
                impossible to reach. If you are using the latest version of the \
                'sampling' library, please contact the library author via github by opening an \
                issue! https://github.com/Pardoxa/sampling/issues");
        e.add_one();
        print!("{} ", e.num())
    }
}


#[derive(Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Rewl<Ensemble, R>{
    ensembles: Vec<RwLock<Ensemble>>,
    map: Vec<usize>,
    worker: Vec<RewlWorker<R>>
}

impl<Ensemble, R> Rewl<Ensemble, R> 
where R: Send + Sync + Rng
{

    pub fn new_from_other_rng<R2>(mut rng: &mut R2, ensembles: Vec<Ensemble>) -> Rewl<Ensemble, R>
    where R2: Rng,
        R: SeedableRng
    {
        let map = (0..ensembles.len()).collect();
        
        let worker = (0..ensembles.len())
            .map(|id| 
                {
                    let r = R::from_rng(&mut rng)
                        .expect("Unable to seed Rng");
                    RewlWorker::new(id, r)
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
        }
    }

    pub fn new(rng: &mut R, ensembles: Vec<Ensemble>) -> Rewl<Ensemble, R>
    where R: Rng + SeedableRng
    {
       Self::new_from_other_rng(rng, ensembles)
    }

    pub fn sweep_test(&mut self)
    where Ensemble: TestEnsemble + Send + Sync,
        R: Send + Sync
    {
        let slice = self.ensembles.as_slice();
        self.worker
            .par_iter_mut()
            .for_each(|w| w.do_work(slice));
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

        let mut rewl = Rewl::new(&mut rng, ensembles);

        for _ in 0..10 {
            rewl.sweep_test();
            println!()
        }
    }
}