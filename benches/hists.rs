use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sampling::histogram::*;
use rand_pcg::Pcg64;
use rand::SeedableRng;
use rand::prelude::*;

pub fn bench_hist(c: &mut Criterion){
    let mut rng = Pcg64::seed_from_u64(black_box(23));
    
    let mut hist = HistU32Fast::new_inclusive(20, 200)
        .unwrap();
    let sampler = rand::distributions::Uniform::new_inclusive(0, 220);
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
    let binning = FastBinningU32::new_inclusive(20, 200);
    let mut hist = GenericHist::new(binning);
    let sampler = rand::distributions::Uniform::new_inclusive(0, 220);
    c.bench_function(
        "new_hist",
        |b| b.iter(|| {
            for _ in 0..1000{
                let _ = hist.count_val(sampler.sample(&mut rng));
            }
        })
    );
}

criterion_group!(benches, bench_hist, bench_hist_new);
criterion_main!(benches);