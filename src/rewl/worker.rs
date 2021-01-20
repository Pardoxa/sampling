use std::unimplemented;
use std::sync::*;

use serde::{Serialize, Deserialize};
use crate::{examples, traits::*};
use rayon::prelude::*;

pub trait TestEnsemble{
    fn num(&self) -> usize;
    fn add_one(&mut self);
}

struct TestEns{
    num: usize
}

impl TestEns{
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
pub struct RewlWorker
{
    id: usize
}

impl RewlWorker {
    pub fn new(id: usize) -> RewlWorker {
        RewlWorker{
            id,
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
pub struct Rewl<Ensemble>{
    ensembles: Vec<RwLock<Ensemble>>,
    map: Vec<usize>,
    worker: Vec<RewlWorker>
}

impl<Ensemble> Rewl<Ensemble> {
    pub fn new(ensembles: Vec<Ensemble>) -> Rewl<Ensemble>
    {
        let map = (0..ensembles.len()).collect();
        let worker = (0..ensembles.len())
            .map(|id| RewlWorker::new(id))
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

    pub fn sweep_test(&mut self)
    where Ensemble: TestEnsemble + Send + Sync
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

    #[test]
    fn proof_of_concept()
    {
        let ensembles = (0..50)
            .map(|num| TestEns::new(num))
            .collect();

        let mut rewl = Rewl::new(ensembles);

        for i in 0..10 {
            rewl.sweep_test();
            println!()
        }
    }
}