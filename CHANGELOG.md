# Changelog

## v0.3.1
* Fixed bug in `push_glue_entry_ignoring` where the missing steps were calculated wrongly
* Fixed bug in `AtomicGenericHist` where get_index mistakenly incremented the index
* Fixed bug in `interval_distance_overlap` where an overlap larger than the bin count could lead to a divide by zero
* Fixed bug in Binning, where it would sometimes truncate instead of saturating at usize::MAX
* Trying to glue empty intervals could lead to a panic -> fixed now

## v0.3.0
* Updating rand to ^0.9.0 -> making required updates to sampling
* Note: Switching to the new rand may lead to other randomness than before, i.e., both should be statistically sound but the same seed may lead to different results in the different library versions

## v0.2.0

Histogram:
* Added Binning trait, which automatically generates histograms
* I might exchange a few histograms with the new underlying binning,
but this would currently be a breaking change for my own simulations,
so I will only do that once my current project is done, if at all
 * renamed `count_index` to `increment_index`
 * renamed `count_multiple_index` to `increment_index_by`

HistogramPartition:
 * in `overlapping_partition`: changed `n` from `usize`to `NonZeroUsize`

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