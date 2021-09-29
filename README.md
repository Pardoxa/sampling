# Scientific Sampling
[![Crate](https://img.shields.io/crates/v/sampling.svg)](https://crates.io/crates/sampling)
[![Docs](https://docs.rs/sampling/badge.svg)](https://docs.rs/sampling/)

Minimal Rust version: 1.55.0

## About

Large-deviation sampling methods (Wang-Landau, Replica-exchange Wang-Landau, 
entropic sampling, Markov-chains), bootstrap resampling, histograms, heat maps and more.
It also allows you to create gnuplot scripts for your heatmaps.

The Documentation of the working branch can be found [here](https://pardoxa.github.io/sampling/sampling/).

## Usage

Add this to your `Cargo.toml`:
```toml
[dependencies]
sampling = "0.1"
# for feature "serde_support" (enabled by default) also use
serde = { version = "1.0", features = ["derive"] }
```
Other features:

`sweep_time_optimization`: Enables minor optimizations, which might 
or might not benefit you for your large-deviation simulation.
This is disabled by default, as most users will not benefit from it.

`sweep_stats`
Also activates feature `sweep_time_optimization`. This is intended for 
testing purposes. You get additional information on how long 
the walkers of `Rewl` take.

`replica_exchange`: enabled by default. Use this, if you want to 
use any of the replica exchange types or methods.

If you want to minimize build time and space requirements upon building,
you can disable default features and only enable what you need.
```toml
sampling = { version = "0.1", default-features = false  }
``` 

# Notes

No warranties whatsoever, but since
I am writing this library for my own scientific simulations,
I do my best to avoid errors.

You can learn more about me and my research on my [homepage](https://www.yfeld.de).

If you notice any bugs, or want to request new features: do not hesitate to
open a new [issue](https://github.com/Pardoxa/sampling/issues) on the repository.

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
