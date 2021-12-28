# Changelog


## v0.2.0 (pending)

WangLandauAdaptive (breaking):
* remove internal field - serialization affected

EntropicSamplingAdaptive (breaking):
* remove internal field - serialization affected

## v0.1.2 (skipped, changes are in v0.2.0)

HistogramFast, HistogramInt and HistogramFloat:
* adding `bin_iter`
* adding `bin_hits_iter`
* adding `increment`
* adding `increment_quiet`

HistogramFast
* adding `equal_range`
* adding `try_add`

## v0.1.1

Fix for Rust nightly. A name became ambiguous and had to be explicitly imported. 