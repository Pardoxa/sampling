use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sampling::{examples::coin_flips::*, *};
use rand_pcg::Pcg64;
use rand::{distr::Uniform, SeedableRng};

pub fn benchmark(c: &mut Criterion){
    let rng = Pcg64::seed_from_u64(23);
    let rng2 = Pcg64::seed_from_u64(23);
    let sequence = CoinFlipSequence::new(10, rng);
    let hist = HistUsizeFast::new_inclusive(0, 10).unwrap();
    let mut wl = WangLandau1T::new(0.000001, sequence, rng2, 500, hist, 1000000).unwrap();
    wl.init_greedy_heuristic(
        |s| Some(s.head_count()),
        None
    ).unwrap();
    c.bench_function(
        "bench",
        |b| b.iter(|| wl.wang_landau_step(|s| Some(s.head_count())))
    );
}

pub fn benchmark2(c: &mut Criterion){
    let rng = Pcg64::seed_from_u64(23);
    let mut sequence = CoinFlipSequence::new(10000, rng);
    let mut step = Vec::with_capacity(128);
    c.bench_function(
        "markov",
        |b| b.iter(|| {
            sequence.m_steps(black_box(128), &mut step);
            sequence.steps_rejected(&step);
            sequence.undo_steps_quiet(&step);
        })
    );
}

pub fn bench_wl_step(c: &mut Criterion){

    use sampling::{*, examples::coin_flips::*};
    use std::num::NonZeroUsize;

    // length of coin flip sequence
    let n = 2000;
    let interval_count = NonZeroUsize::new(3).unwrap();
    let step_size = 10;

    // create histogram. The result of our `energy` (number of heads) can be anything between 0 and n
    let hist = HistUsizeFast::new_inclusive(0, n).unwrap();

    // now the overlapping histograms for sampling
    // lets create 3 histograms. The parameter Overlap should be larger than 0. Normally, 1 is sufficient
    let hist_list = hist.overlapping_partition(interval_count, 1).unwrap();
    // alternativly you could also create the histograms in the desired interval. 
    // Just make sure, that they overlap

    // create rng to seed all other rngs
    let mut rng = Pcg64::seed_from_u64(834628956578);

    // now create ensembles (could be combined with wl creation)
    // note: You could also create one ensemble and clone it instead of creating different ones
    let ensembles: Vec<_> = (0..interval_count.get()).map(|_| {
        CoinFlipSequence::new(
            n,
            Pcg64::from_rng(&mut rng)
        )
    }).collect();
    
    // Now the Wang Landau simulation. First create the struct 
    // (here as Vector, since we want to use 3 overlapping intervals)
    let mut wl_list: Vec<_> = ensembles.into_iter()
        .zip(hist_list)
        .map(|(ensemble, histogram)| {
            WangLandau1T::new(
                0.0000000000000000001, // arbitrary threshold for `log_f`(see paper), 
                         // you have to try what is good for your model
                ensemble,
                Pcg64::from_rng(&mut rng),
                step_size,  // stepsize 1 is sufficient for this problem
                histogram,
                100 // every 100 steps: check if WL can refine factor f
            ).unwrap()
        }).collect();
    
    // Now we have to initialize the wl with a valid state
    // as the simulation has to start in the interval one wants to measure.
    // Since the energy landscape is quite simple, here a greedy approach is good enough.
    
    wl_list.iter_mut()
        .for_each(|wl|{
            wl.init_greedy_heuristic(
                |coin_seq| Some(coin_seq.head_count()),
                Some(10_000) // if no valid state is found after 10_000 
                             // this returns an Err. If you do not want a step limit,
                             // you can use None here
            ).expect("Unable to find valid state within 10_000 steps!");
        });

    

    c.bench_function(
        "wl_step",
        |b| b.iter(|| {
            for wl in wl_list.iter_mut(){
                wl.wang_landau_step(|e| Some(e.head_count()))
            }
        })
    );
}

pub fn bench_wl_step_acc(c: &mut Criterion){

    use sampling::{*, examples::coin_flips::*};
    use std::num::NonZeroUsize;

    // length of coin flip sequence
    let n = 2000;
    let interval_count = NonZeroUsize::new(3).unwrap();
    let step_size = 10;

    // create histogram. The result of our `energy` (number of heads) can be anything between 0 and n
    let hist = HistUsizeFast::new_inclusive(0, n).unwrap();

    // now the overlapping histograms for sampling
    // lets create 3 histograms. The parameter Overlap should be larger than 0. Normally, 1 is sufficient
    let hist_list = hist.overlapping_partition(interval_count, 1).unwrap();
    // alternativly you could also create the histograms in the desired interval. 
    // Just make sure, that they overlap

    // create rng to seed all other rngs
    let mut rng = Pcg64::seed_from_u64(834628956578);

    // now create ensembles (could be combined with wl creation)
    // note: You could also create one ensemble and clone it instead of creating different ones
    let ensembles: Vec<_> = (0..interval_count.get()).map(|_| {
        CoinFlipSequence::new(
            n,
            Pcg64::from_rng(&mut rng)
        )
    }).collect();
    
    // Now the Wang Landau simulation. First create the struct 
    // (here as Vector, since we want to use 3 overlapping intervals)
    let mut wl_list: Vec<_> = ensembles.into_iter()
        .zip(hist_list)
        .map(|(ensemble, histogram)| {
            WangLandau1T::new(
                0.0000000000000000001, // arbitrary threshold for `log_f`(see paper), 
                         // you have to try what is good for your model
                ensemble,
                Pcg64::from_rng(&mut rng),
                step_size,  // stepsize 1 is sufficient for this problem
                histogram,
                100 // every 100 steps: check if WL can refine factor f
            ).unwrap()
        }).collect();
    
    // Now we have to initialize the wl with a valid state
    // as the simulation has to start in the interval one wants to measure.
    // Since the energy landscape is quite simple, here a greedy approach is good enough.
    
    wl_list.iter_mut()
        .for_each(|wl|{
            wl.init_greedy_heuristic(
                |coin_seq| Some(coin_seq.head_count()),
                Some(10_000) // if no valid state is found after 10_000 
                             // this returns an Err. If you do not want a step limit,
                             // you can use None here
            ).expect("Unable to find valid state within 10_000 steps!");
        });

    

    c.bench_function(
        "wl_step_acc",
        |b| b.iter(|| {
            for wl in wl_list.iter_mut(){
                wl.wang_landau_step_acc(|ensemble, step, energy|
                    {
                        ensemble.update_head_count(step, energy);
                    }
                );
            }
        })
    );
}

pub fn bench_hists(c: &mut Criterion){
    use sampling::*;
    use rand::prelude::*;
    use rand_pcg::Pcg64;

    let max_val = 100;
    let mut hist = HistI16::new_inclusive(0, max_val, (max_val+1) as usize)
        .unwrap();

    let uniform = Uniform::new_inclusive(0, max_val)
        .unwrap();
    let mut rng = Pcg64::seed_from_u64(23894623987612);

    c.bench_function(
        "HistI16", 
        |b|
        {
            b.iter(
                || {
                    for _ in 0..100{
                        let num = uniform.sample(&mut rng);
                        hist.increment_quiet(num);
                    }
                }
            );
        }
    );

    let mut hist = BinningI16::new_inclusive(
        0, 
        max_val, 
        1
    ).unwrap()
    .to_generic_hist();

    let mut rng = Pcg64::seed_from_u64(23894623987612);

    c.bench_function(
        "GenericHistBinning", 
        |b|
        {
            b.iter(
                || {
                    for _ in 0..100{
                        let num = uniform.sample(&mut rng);
                        let _ = hist.count_val(num);
                    }
                }
            );
        }
    );

    let mut hist = FastBinningI16::new_inclusive(
        0, 
        max_val,
    ).to_generic_hist();

    let mut rng = Pcg64::seed_from_u64(23894623987612);

    c.bench_function(
        "GenericHistFastBinning", 
        |b|
        {
            b.iter(
                || {
                    for _ in 0..100{
                        let num = uniform.sample(&mut rng);
                        let _ = hist.count_val(num);
                    }
                }
            );
        }
    );

    let mut hist = HistI16Fast::new_inclusive(
        0, 
        max_val,
    ).unwrap();

    let mut rng = Pcg64::seed_from_u64(23894623987612);

    c.bench_function(
        "HistI16Fast", 
        |b|
        {
            b.iter(
                || {
                    for _ in 0..100{
                        let num = uniform.sample(&mut rng);
                        hist.increment_quiet(num);
                    }
                }
            );
        }
    );

}

criterion_group!(benches, benchmark, benchmark2, bench_wl_step, bench_wl_step_acc, bench_hists);
criterion_main!(benches);