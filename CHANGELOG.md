# Changelog


## v0.2.0 (pending)

Histogram:
* Added Binning trait, which automatically generates histograms
* I might exchange a few histograms with the new underlying binning,
but this would currently be a breaking change for my own simulations,
so I will only do that once my current project is done, if at all
 * renamed `count_index` to `increment_index`
 * renamed `count_multiple_index` to `increment_index_by`

Glue:
New way to glue the simulations, removed the old one, see examples

WangLandauAdaptive (breaking):
* remove internal field - serialization affected

EntropicSamplingAdaptive (breaking):
* remove internal field - serialization affected

MarkovChain:
* Added functions steps_accepted and steps_rejected - they can be used to create your own statistics 

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