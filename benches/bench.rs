use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sampling::{examples::coin_flips::*, *};
use rand_pcg::Pcg64;
use rand::SeedableRng;

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
            sequence.undo_steps_quiet(&mut step);
        })
    );
}

criterion_group!(benches, benchmark, benchmark2);
criterion_main!(benches);