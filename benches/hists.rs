use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sampling::histogram::*;
use rand_pcg::Pcg64;
use rand::SeedableRng;
use rand::prelude::*;

pub fn bench_hist(c: &mut Criterion){
    let mut rng = Pcg64::seed_from_u64(black_box(23));
    
    let mut hist = HistU32Fast::new_inclusive(20, 20000)
        .unwrap();
    let sampler = rand::distr::Uniform::new_inclusive(0, 22000)
        .unwrap();
    c.bench_function(
        "old_hist",
        |b| b.iter(|| {
            for _ in 0..1000{
                hist.increment_quiet(sampler.sample(&mut rng));
            }
        })
    );
}

pub fn bench_hist_new(c: &mut Criterion){
    let mut rng = Pcg64::seed_from_u64(black_box(23));
    let binning = FastBinningU32::new_inclusive(20, 20000);
    let mut hist = GenericHist::new(binning);
    let sampler = rand::distr::Uniform::new_inclusive(0, 22000)
        .unwrap();
    c.bench_function(
        "new_hist",
        |b| b.iter(|| {
            for _ in 0..1000{
                let _ = hist.count_val(sampler.sample(&mut rng));
            }
        })
    );
}

pub fn bench_hist_new_multi(c: &mut Criterion){
    let mut rng = Pcg64::seed_from_u64(black_box(23));
    let binning = BinningU32::new_inclusive(21, 200, 2)
        .unwrap();
    let mut hist = GenericHist::new(binning);
    let sampler = rand::distr::Uniform::new_inclusive(0, 220)
        .unwrap();
    c.bench_function(
        "new_hist_multi",
        |b| b.iter(|| {
            for _ in 0..1000{
                let _ = hist.count_val(sampler.sample(&mut rng));
            }
        })
    );
}

pub fn bench_hist_old_multi(c: &mut Criterion){
    let mut rng = Pcg64::seed_from_u64(black_box(23));
    #[allow(deprecated)]
    let mut hist = HistU32::new_inclusive(21,200, 90)
        .unwrap();
    let sampler = rand::distr::Uniform::new_inclusive(0, 220)
        .unwrap();
    c.bench_function(
        "old_hist_multi",
        |b| b.iter(|| {
            for _ in 0..1000{
                let _ = hist.count_val(sampler.sample(&mut rng));
            }
        })
    );
}

criterion_group!(benches, bench_hist_old_multi, bench_hist_new_multi, bench_hist, bench_hist_new);
criterion_main!(benches);