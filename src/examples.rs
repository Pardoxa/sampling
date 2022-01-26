//! Contains examples

/// # Example Coin flips
/// 
/// Lets assume we do n Coinflips and want to measure the probability for the number of times,
/// this results in Head. This means, the number of times the coin flip returned head is the `energy`
///
/// Of cause, for this example there is a analytic solution.
/// 
/// For the implementation of the coin flip sequence and Markov chain of it, please look in the [source code](../../../src/sampling/examples/coin_flips.rs.html)
/// 
/// Now A detailed example for large deviation simulations with **comparison to analytical results**
///
/// The files created in this example can be found below
/// ```
///
/// use rand::SeedableRng;
/// use rand_pcg::Pcg64;
/// use sampling::{*, examples::coin_flips::*};
/// use std::fs::File;
/// use std::io::{BufWriter, Write};
/// use statrs::distribution::{Binomial, Discrete};
///
/// // length of coin flip sequence
/// let n = 20;
/// let interval_count = 3;
/// 
/// // create histogram. The result of our `energy` (number of heads) can be anything between 0 and n
/// let hist = HistUsizeFast::new_inclusive(0, n).unwrap();
/// 
/// // now the overlapping histograms for sampling
/// // lets create 3 histograms. The parameter Overlap should be larger than 0. Normally, 1 is sufficient
/// let hist_list = hist.overlapping_partition(interval_count, 1).unwrap();
/// // alternativly you could also create the histograms in the desired interval. 
/// // Just make sure, that they overlap
/// 
/// // create rng to seed all other rngs
/// let mut rng = Pcg64::seed_from_u64(834628956578);
/// 
/// // now create ensembles (could be combined with wl creation)
/// // note: You could also create one ensemble and clone it instead of creating different ones
/// let ensembles: Vec<_> = (0..interval_count).map(|_| {
///     CoinFlipSequence::new(
///         n,
///         Pcg64::from_rng(&mut rng).unwrap()
///     )
/// }).collect();
/// 
/// // Now the Wang Landau simulation. First create the struct 
/// // (here as Vector, since we want to use 3 overlapping intervals)
/// let mut wl_list: Vec<_> = ensembles.into_iter()
///     .zip(hist_list.into_iter())
///     .map(|(ensemble, histogram)| {
///         WangLandau1T::new(
///             0.00001, // arbitrary threshold for `log_f`(see paper), 
///                      // you have to try what is good for your model
///             ensemble,
///             Pcg64::from_rng(&mut rng).unwrap(),
///             1,  // stepsize 1 is sufficient for this problem
///             histogram,
///             100 // every 100 steps: check if WL can refine factor f
///         ).unwrap()
///     }).collect();
/// 
/// // Now we have to initialize the wl with a valid state
/// // as the simulation has to start in the interval one wants to measure.
/// // Since the energy landscape is quite simple, here a greedy approach is good enough.
/// 
/// wl_list.iter_mut()
///     .for_each(|wl|{
///         wl.init_greedy_heuristic(
///             |coin_seq| Some(coin_seq.head_count()),
///             Some(10_000) // if no valid state is found after 10_000 
///                          // this returns an Err. If you do not want a step limit,
///                          // you can use None here
///         ).expect("Unable to find valid state within 10_000 steps!");
///     });
/// 
/// // Now our ensemble is initialized. Time for the Wang Landau Simulation. 
/// // You can do that in different ways.
/// // I will show this by doing it differently for our three intervals
/// 
/// // First, the simplest one. Just simulate until the threshold for `log_f` is reached
/// wl_list[0].wang_landau_convergence(
///     |coin_seq| Some(coin_seq.head_count())
/// );
/// 
/// // Secondly, I only have a limited amount of time.
/// // Lets say, I have 1 minute at most.
/// let start_time = std::time::Instant::now();
/// wl_list[1].wang_landau_while(
///     |coin_seq| Some(coin_seq.head_count()),
///     |_| start_time.elapsed().as_secs() <= 60
/// );
/// 
/// // Or lets say, I want to limit the number of steps to 100_000
/// wl_list[2].wang_landau_while(
///     |coin_seq| Some(coin_seq.head_count()),
///     |state| state.step_counter() <= 100_000 
/// );
/// 
/// // Now, lets see if our last two simulations did indeed finish:
/// // This one did
/// assert!(wl_list[1].is_finished());
/// // This simulation did not finish
/// assert!(!wl_list[2].is_finished());
/// 
/// // If a simulation did not finish, you could, e.g., store the state (`wl_list[2]`) using serde.
/// // Then you could continue the simulation later on.
/// // I recommend the crate `bincode` for storing
/// 
/// // lets resume the simulation for now
/// wl_list[2].wang_landau_convergence(
///     |coin_seq| Some(coin_seq.head_count())
/// );
/// // it finished
/// assert!(wl_list[2].is_finished());
/// 
/// // Since our simulations did all finish, lets see what our distribution looks like
/// // Lets glue them together. We use our original histogram for that.
/// let glued = glue_wl(
///     &wl_list,
///     &hist
/// ).expect("Unable to glue results. Look at error message");
/// 
/// // now, lets print our result
/// glued.write(std::io::stdout()).unwrap();
/// 
/// // or store it into a file
/// let file = File::create("coin_flip_log_density.dat").unwrap();
/// let buf = BufWriter::new(file);
/// glued.write(buf).unwrap();
/// 
/// // now, lets check if our results are actually any good.
/// // lets compare that to the analytical result
/// 
/// // Since the library I am going to use lets me directly calculate the natural
/// // logaritm of the probability, I first convert the base of our own results:
/// let log10_prob = glued.glued_log10_probability;
/// let ln_prob: Vec<_> = log10_prob.iter()
///                         .map(|&val| val / std::f64::consts::LOG10_E)
///                         .collect();
/// 
/// // Then create the `true` results:
/// let binomial = Binomial::new(0.5, n as u64).unwrap();
/// 
/// let ln_prob_true: Vec<_> = (0..=n)
///     .map(|k| binomial.ln_pmf(k as u64))
///     .collect();
/// 
/// // lets write that in a file, so we can use gnuplot to plot the result
/// let comp_file = File::create("coin_flip_compare.dat").unwrap();
/// let mut buf = BufWriter::new(comp_file);
/// 
/// // lets also calculate the maximum difference between the two solutions
/// let mut max_ln_dif = std::f64::NEG_INFINITY;
/// let mut max_dif = std::f64::NEG_INFINITY;
/// 
/// writeln!(buf, "#head_count Numeric_ln_prob Analytic_ln_prob ln_dif dif").unwrap();
/// for (index, (numeric, analytic)) in ln_prob.iter().zip(ln_prob_true.iter()).enumerate()
/// {
///     let ln_dif = numeric - analytic;
///     max_ln_dif = ln_dif.abs().max(max_ln_dif);
///     let dif = numeric.exp() - analytic.exp();
///     max_dif = dif.abs().max(max_dif);
///     writeln!(buf, "{} {:e} {:e} {:e} {:e}", index, numeric, analytic, ln_dif, dif).unwrap();
/// }
/// 
/// println!("Max_ln_dif = {}", max_ln_dif);
/// println!("Max_dif = {}", max_dif);
/// 
/// // in this case, the max difference of the natural logarithms 
/// // of the probabilities is smaller than 0.03
/// assert!(max_ln_dif < 0.03);
/// // and the max absolut difference is smaller than 0.0009
/// assert!(max_dif < 0.0009);
/// 
/// // But we can do better. Lets refine the results with entropic sampling
/// // first, convert the wl simulations in entropic sampling simulations
/// let mut entropic_list: Vec<_> = wl_list
///     .into_iter()
///     .map(|wl| EntropicSampling::from_wl(wl).unwrap())
///     .collect();
/// 
/// 
/// // Now, while doing that, lets also create a heatmap.
/// // Lets say, we want to see, how the number of times `Head` occured in the sequence 
/// // correlates to the maximum number of `Heads` in a row in that sequence.
/// 
/// // In this case, the heatmap is symetric and we already have a histogram of correct sice
/// let mut heatmap = HeatmapU::new(
///     hist.clone(),
///     hist.clone()
/// );
/// 
/// entropic_list.iter_mut()
///     .for_each(|entr|{
///         entr.entropic_sampling(
///             |coin_seq| Some(coin_seq.head_count()),
///             |state| {
///                 let head_count = *state.energy();
///                 let heads_in_row = state.ensemble().max_heads_in_a_row();
///                 heatmap.count(head_count, heads_in_row)
///                     .expect("Value outside heatmap?");
///             }
///         )
///     });
/// 
/// // Now, lets see our refined results:
/// let glued = glue_entropic(
///     &entropic_list,
///     &hist
/// ).expect("Unable to glue results. Look at error message");
/// 
/// // lets store our result
/// let file = File::create("coin_flip_log_density_entropic.dat").unwrap();
/// let buf = BufWriter::new(file);
/// glued.write(buf).unwrap();
/// 
/// // now, lets compare with the analytical results again
/// // Again, calculate to base e
/// let ln_prob: Vec<_> = glued.glued_log10_probability
///     .iter()
///     .map(|&val| val / std::f64::consts::LOG10_E)
///     .collect();
/// 
/// 
/// // lets write that in a file, so we can use gnuplot to plot the result
/// let comp_file = File::create("coin_flip_compare_entr.dat").unwrap();
/// let mut buf = BufWriter::new(comp_file);
/// 
/// // lets also calculate the maximum difference between the two solutions
/// let mut max_ln_dif = std::f64::NEG_INFINITY;
/// let mut max_dif = std::f64::NEG_INFINITY;
/// 
/// writeln!(buf, "#head_count Numeric_ln_prob Analytic_ln_prob ln_dif dif").unwrap();
/// for (index, (numeric, analytic)) in ln_prob.iter().zip(ln_prob_true.iter()).enumerate()
/// {
///     let ln_dif = numeric - analytic;
///     max_ln_dif = ln_dif.abs().max(max_ln_dif);
///     let dif = numeric.exp() - analytic.exp();
///     max_dif = dif.abs().max(max_dif);
///     writeln!(buf, "{} {:e} {:e} {:e} {:e}", index, numeric, analytic, ln_dif, dif).unwrap();
/// }
/// 
/// println!("Max_ln_dif = {}", max_ln_dif);
/// println!("Max_dif = {}", max_dif);
/// 
/// // in this case, the max difference of the natural logarithms 
/// //of the probabilities is smaller than 0.026
/// assert!(max_ln_dif < 0.026);
/// // and the max absolut difference is smaller than 0.0007
/// assert!(max_dif < 0.0007);
/// 
/// // That would be the final result for our probability 
/// // density than. As you can see, it is very very 
/// // close to the analytical result.
/// 
/// // Now, lets see, how our heatmap looks:
/// let mut settings = GnuplotSettings::new();
/// settings.x_label("#Heads")
///     .y_label("Max heads in row");
/// 
/// // lets normalize coloumwise
/// // This way, the scale of our heatmap tells us the conditional probability
/// // P(Number of heads in a rom | number of heads) of how many heads in a row were
/// // part of that sequence given the total number of heads that occured in the sequence
/// let heatmap = heatmap.heatmap_normalized_columns();
/// 
/// // now create gnuplot file
/// let file = File::create("coin_heatmap.gp").unwrap();
/// let buf = BufWriter::new(file);
/// heatmap.gnuplot(
///     buf,
///     "heatmap_coin_flips",
///     settings
/// ).unwrap();
/// 
/// // now you can use gnuplot to plot the heatmap
/// ```
/// # Created files:
/// * `coin_flip_log_density.dat`
/// ```txt 
/// #bin_left bin_right glued_log_density curve_0 curve_1 curve_2
/// #total_steps 3300000
/// #total_steps_accepted 1892678
/// #total_steps_rejected 1407322
/// #total_acception_fraction 5.735387878787879e-1
/// #total_rejection_fraction 4.2646121212121213e-1
/// 0 1 -6.031606841072606e0 -5.2779314834625595e0 NONE NONE
/// 1 2 -4.728485280490379e0 -3.9748099228803326e0 NONE NONE
/// 2 3 -3.7500990622715635e0 -2.9964237046615168e0 NONE NONE
/// 3 4 -2.9668930048460895e0 -2.2132176472360428e0 NONE NONE
/// 4 5 -2.339721588163688e0 -1.5860462305536416e0 NONE NONE
/// 5 6 -1.8365244111917054e0 -1.0820006622583023e0 -1.083697444905016e0 NONE
/// 6 7 -1.4371802248730712e0 -6.836525846019086e-1 -6.83357149924141e-1 NONE
/// 7 8 -1.1351008012679982e0 -3.8187173971763855e-1 -3.809791475982654e-1 NONE
/// 8 9 -9.233969878866752e-1 -1.7037710459167243e-1 -1.6906615596158522e-1 NONE
/// 9 10 -7.962413727681732e-1 -4.269821376793459e-2 -4.243381654831898e-2 NONE
/// 10 11 -7.528173091050847e-1 0e0 -1.0665900001297264e-3 3.6407355150146854e-3
/// 11 12 -7.942713970200067e-1 NONE -4.26721996981397e-2 -3.8519879121780974e-2
/// 12 13 -9.171173964543223e-1 NONE -1.6557876219227494e-1 -1.61305315496277e-1
/// 13 14 -1.1271626249139948e0 NONE -3.736972983947891e-1 -3.732772362131079e-1
/// 14 15 -1.430004495200477e0 NONE -6.750424918819155e-1 -6.776157832989455e-1
/// 15 16 -1.8307880991746777e0 NONE -1.0716228097885552e0 -1.0826026733407075e0
/// 16 17 -2.3447129767461368e0 NONE NONE -1.5910376191360902e0
/// 17 18 -2.974429977113245e0 NONE NONE -2.220754619503199e0
/// 18 19 -3.7503197785367197e0 NONE NONE -2.9966444209266734e0
/// 19 20 -4.732341369591395e0 NONE NONE -3.9786660119813484e0
/// 20 21 -6.014059503852461e0 NONE NONE -5.260384146242414e0
/// ```
/// * `coin_flip_compare.dat`
/// ```dat
/// #head_count Numeric_ln_prob Analytic_ln_prob ln_dif dif
/// 0 -1.388828799905469e1 -1.3862943611198906e1 -2.5344387855783523e-2 -2.3866572408958168e-8
/// 1 -1.0887739719298917e1 -1.0867211337644916e1 -2.0528381654001393e-2 -3.875562455014197e-7
/// 2 -8.634922198037453e0 -8.615919539038421e0 -1.9002658999031752e-2 -3.4107369181183742e-6
/// 3 -6.831523605466917e0 -6.824160069810366e0 -7.363535656550901e-3 -7.976150535885006e-6
/// 4 -5.387408050662062e0 -5.377241086874042e0 -1.0166963788019956e-2 -4.673898610979728e-5
/// 5 -4.228753732129688e0 -4.214090277068356e0 -1.4663455061331376e-2 -2.152285704062097e-4
/// 6 -3.309229761738564e0 -3.2977995451941995e0 -1.1430216544364491e-2 -4.201057612779821e-4
/// 7 -2.6136661840452895e0 -2.6046523646342594e0 -9.013819411030077e-3 -6.633868338237203e-4
/// 8 -2.126200139223462e0 -2.119144548852554e0 -7.0555903709079715e-3 -8.446355834740987e-4
/// 9 -1.8334135153611106e0 -1.8314624764007785e0 -1.9510389603321077e-3 -3.122110722084126e-4
/// 10 -1.7334259136932588e0 -1.736152296596451e0 2.7263829031922704e-3 4.810360764700705e-4
/// 11 -1.8288774785698227e0 -1.8314624764007768e0 2.5849978309540056e-3 4.145983618321636e-4
/// 12 -2.1117408456012328e0 -2.1191445488525558e0 7.403703251323002e-3 8.927398170218287e-4
/// 13 -2.5953878575070033e0 -2.6046523646342594e0 9.264507127256127e-3 6.880967171132152e-4
/// 14 -3.2927070335630937e0 -3.2977995451942013e0 5.092511631107577e-3 1.8872184723020546e-4
/// 15 -4.215545385590517e0 -4.214090277068356e0 -1.4551085221610194e-3 -2.1499249324725273e-5
/// 16 -5.398901147605349e0 -5.377241086874038e0 -2.1660060731310438e-2 -9.900533675925687e-5
/// 17 -6.848878125455579e0 -6.824160069810366e0 -2.47180556452129e-2 -2.6543784456516638e-5
/// 18 -8.635430416019382e0 -8.61591953903842e0 -1.9510876980962877e-2 -3.5010687071566015e-6
/// 19 -1.0896618692580171e1 -1.0867211337644916e1 -2.940735493525537e-2 -5.52733731038154e-7
/// 20 -1.3847883761949843e1 -1.3862943611198906e1 1.5059849249063006e-2 1.4470882595462752e-8
/// ```
/// * `coin_flip_log_density_entropic.dat`
/// * If you want to plot with gnuplot:
/// ```gp
/// set format y "10^{%.0f}"
/// set ylabel 'P(Heads)'
/// set xlabel '#Heads'
/// p "coin_flip_log_density_entropic.dat" u 1:3 t 'P(E), normalized',\
///    for[i=4:6] "" u 1:i t "glued not normalized overlapping interval"
/// ```
/// ```csv
/// #bin_left bin_right glued_log_density curve_0 curve_1 curve_2
/// #total_steps 6600000
/// #total_steps_accepted 3789779
/// #total_steps_rejected 2810221
/// #total_acception_fraction 5.742089393939394e-1
/// #total_rejection_fraction 4.257910606060606e-1
/// 0 1 -6.026918605089148e0 -5.27395131583382e0 NONE NONE
/// 1 2 -4.7282564130642415e0 -3.9752891238089134e0 NONE NONE
/// 2 3 -3.740764114870072e0 -2.987796825614744e0 NONE NONE
/// 3 4 -2.9571447347511652e0 -2.204177445495837e0 NONE NONE
/// 4 5 -2.3331262146639107e0 -1.5801589254085826e0 NONE NONE
/// 5 6 -1.822752729218271e0 -1.072607725281621e0 -1.0669631546442648e0 NONE
/// 6 7 -1.4288440916787124e0 -6.757499681532266e-1 -6.76003636693542e-1 NONE
/// 7 8 -1.1277454047397892e0 -3.750318346032344e-1 -3.745243963656879e-1 NONE
/// 8 9 -9.191559949607412e-1 -1.6656676951174632e-1 -1.6581064189908012e-1 NONE
/// 9 10 -7.959284032103325e-1 -4.211860403626844e-2 -4.3803623873740705e-2 NONE
/// 10 11 -7.547205033527027e-1 0e0 -4.969448109781283e-3 -2.901941823427734e-4
/// 11 12 -7.971134272615988e-1 NONE -4.4924054052993156e-2 -4.336822195954859e-2
/// 12 13 -9.198413168594478e-1 NONE -1.6588986141874607e-1 -1.6785819378949363e-1
/// 13 14 -1.1307079438518484e0 NONE -3.7892725389989756e-1 -3.76554055293143e-1
/// 14 15 -1.4352206678153943e0 NONE -6.801668342346883e-1 -6.843399228854441e-1
/// 15 16 -1.836086569993232e0 NONE -1.0818858489348369e0 -1.084352712540971e0
/// 16 17 -2.3462415423683174e0 NONE NONE -1.5932742531129893e0
/// 17 18 -2.971577175460559e0 NONE NONE -2.218609886205231e0
/// 18 19 -3.7473120822317787e0 NONE NONE -2.9943447929764506e0
/// 19 20 -4.723849161266183e0 NONE NONE -3.970881872010855e0
/// 20 21 -6.016589039252039e0 NONE NONE -5.263621749996711e0
/// ```
/// * `coin_flip_compare_entr.dat`
/// * gnuplot for comparing:
/// ```gp
/// set format y "e^{%.0f}"
/// set ylabel 'P(Heads)'
/// set xlabel '#Heads'
/// p "coin_flip_compare_entr.dat" u 1:2 t "numeric results", "" u 1:3 t "analytic results"
/// ```
/// ```txt
/// #head_count Numeric_ln_prob Analytic_ln_prob ln_dif dif
/// 0 -1.387749293676674e1 -1.3862943611198906e1 -1.4549325567834615e-2 -1.3774867607233972e-8
/// 1 -1.088721273257522e1 -1.0867211337644916e1 -2.0001394930304173e-2 -3.777064132904934e-7
/// 2 -8.613427687306894e0 -8.615919539038421e0 2.491851731527106e-3 4.5208187593923066e-7
/// 3 -6.8090773840638645e0 -6.824160069810366e0 1.5082685746501845e-2 1.652201075832624e-5
/// 4 -5.372221641958746e0 -5.377241086874042e0 5.019444915295601e-3 2.3250911075241125e-5
/// 5 -4.197043262512203e0 -4.214090277068356e0 1.7047014556153428e-2 2.542138155917483e-4
/// 6 -3.290035105712021e0 -3.2977995451941995e0 7.764439482178531e-3 2.8812509235658784e-4
/// 7 -2.5967297576463753e0 -2.6046523646342594e0 7.922606987884162e-3 5.880353998786031e-4
/// 8 -2.116434892132713e0 -2.119144548852554e0 2.709656719841025e-3 3.2596428483325224e-4
/// 9 -1.832692876322666e0 -1.8314624764007785e0 -1.2303999218874484e-3 -1.9696320250631172e-4
/// 10 -1.7378081803968959e0 -1.736152296596451e0 -1.6558838004447907e-3 -2.91520415518165e-4
/// 11 -1.835421495037951e0 -1.8314624764007768e0 -3.959018637174294e-3 -6.328985381396646e-4
/// 12 -2.1180129041205773e0 -2.1191445488525558e0 1.131644731978465e-3 1.3602636066446794e-4
/// 13 -2.6035512560432146e0 -2.6046523646342594e0 1.1011085910448415e-3 8.144850674679516e-5
/// 14 -3.304717714868686e0 -3.2977995451942013e0 -6.918169674484886e-3 -2.5484356336342995e-4
/// 15 -4.2277455655129845e0 -4.214090277068356e0 -1.3655288444628155e-2 -2.0053163313986377e-4
/// 16 -5.4024208000206455e0 -5.377241086874038e0 -2.5179713146607163e-2 -1.1489163608931121e-4
/// 17 -6.842309306896834e0 -6.824160069810366e0 -1.8149237086467984e-2 -1.9553667043482144e-5
/// 18 -8.62850493934337e0 -8.61591953903842e0 -1.2585400304951477e-2 -2.266160694642524e-6
/// 19 -1.0877064660283938e1 -1.0867211337644916e1 -9.853322639022721e-3 -1.8701434523338205e-7
/// 20 -1.3853708232453112e1 -1.3862943611198906e1 9.235378745794165e-3 8.848339504323986e-9
/// ```
/// * `coin_heatmap.gp`
/// If you want to see how it looks like, you can copy the file and use `gnuplot coin_heatmap.gp`
/// ```gp
/// set t pdf
/// set output "heatmap_coin_flips.pdf"
/// set xlabel "#Heads"
/// set ylabel "Max heads in row"
/// set xrange[-0.5:20.5]
/// set yrange[-0.5:20.5]
/// set palette model HSV
/// set palette negative defined  ( 0 0 1 0, 2.8 0.4 0.6 0.8, 5.5 0.83 0 1 )
/// set view map
/// set rmargin screen 0.8125
/// set lmargin screen 0.175
/// $data << EOD
/// 1e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0
/// 0e0 1e0 8.942007280084902e-1 7.126507397021373e-1 4.9517412091147733e-1 2.7688123477165416e-1 1.2722327984770193e-1 4.4177045183096356e-2 1.0028102647039738e-2 1.153466443696734e-3 6.05371664570287e-5 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0
/// 0e0 0e0 1.0579927199150987e-1 2.713958719368999e-1 4.4540817854255454e-1 5.843583220621195e-1 6.111009469181893e-1 5.423656574767444e-1 4.069492904455992e-1 2.552988938810872e-1 1.2900470171992817e-1 5.0431631137845115e-2 1.265593470242199e-2 1.4620914966775231e-3 2.0244759137978155e-5 0e0 0e0 0e0 0e0 0e0 0e0
/// 0e0 0e0 0e0 1.595338836096275e-2 5.5904340795895756e-2 1.2220762796693344e-1 2.1528419907533317e-1 3.115537295754266e-1 3.9456016612008765e-1 4.327513952410694e-1 4.1325696681890645e-1 3.289708806281932e-1 2.0963058760775158e-1 9.928609601403608e-2 3.0731544371450842e-2 4.150637780927313e-3 0e0 0e0 0e0 0e0 0e0
/// 0e0 0e0 0e0 0e0 3.5133597500723633e-3 1.5532302403820331e-2 4.0749622963384184e-2 8.303143538227467e-2 1.426646101319277e-1 2.137791388793746e-1 2.7967161950372976e-1 3.250799083884932e-1 3.3032744400440817e-1 2.790729331572101e-1 1.8010749967102266e-1 7.551123709252885e-2 1.5334414349488519e-2 0e0 0e0 0e0 0e0
/// 0e0 0e0 0e0 0e0 0e0 1.020512795472634e-3 5.360100872747051e-3 1.6354526486899033e-2 3.6483525259937005e-2 7.120262728426652e-2 1.1868647801491905e-1 1.764729569878942e-1 2.335334463896619e-1 2.6976596452663526e-1 2.620835906104807e-1 1.994381453735574e-1 9.346785978820091e-2 1.7855904240034085e-2 0e0 0e0 0e0
/// 0e0 0e0 0e0 0e0 0e0 0e0 2.818503226444483e-4 2.354060630690012e-3 7.886710892146733e-3 2.0223440049966755e-2 4.2399558751320045e-2 7.696876651649762e-2 1.2222160717790269e-1 1.7494681011968982e-1 2.2017693919486592e-1 2.2753593844907877e-1 1.7919844653974737e-1 8.109473934541563e-2 5.9234312317369125e-3 0e0 0e0
/// 0e0 0e0 0e0 0e0 0e0 0e0 0e0 1.6354526486899034e-4 1.3627038440228215e-3 4.780085829992143e-3 1.296504314954698e-2 2.89431958321798e-2 5.689886826253893e-2 9.565103405160678e-2 1.4196131226528733e-1 1.8725450496051832e-1 2.0002802634476408e-1 1.470981678375727e-1 4.815016591550691e-2 0e0 0e0
/// 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 6.489065923918197e-5 7.555457054782101e-4 3.2690069886795498e-3 9.845720182216293e-3 2.3223514374424443e-2 4.6333175360229094e-2 8.246196515876952e-2 1.280471755416076e-1 1.6467479430664825e-1 1.6756012247445973e-1 9.21400624040414e-2 0e0 0e0
/// 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 5.540668506840207e-5 6.59182479198757e-4 2.6980092114866736e-3 8.459095918398156e-3 2.1109576195133758e-2 4.497879361480297e-2 8.13778092731322e-2 1.2503753528316616e-1 1.6224893231205223e-1 1.4450993016690605e-1 0e0 0e0
/// 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 2.6905407314234977e-5 5.184607253416555e-4 2.4255111437643732e-3 8.742298811168362e-3 2.2820904738285876e-2 4.859789431058919e-2 8.993453846615819e-2 1.289251776176934e-1 1.473428755386063e-1 9.912165730941529e-2 0e0
/// 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 7.047038985226386e-5 5.334111643963144e-4 2.8233490970324586e-3 1.0051522912006155e-2 2.7029763109941284e-2 5.953596380597762e-2 9.816783757270682e-2 1.2446139369025803e-1 1.0074531974412645e-1 0e0
/// 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 9.057925433144962e-5 7.007955794419852e-4 3.5731999878531443e-3 1.3302287912532901e-2 3.629411646948131e-2 7.41287567256909e-2 1.094943291565549e-1 1.0805180070032669e-1 0e0
/// 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 1.0587559113871718e-4 9.211365407780061e-4 5.724843085644867e-3 2.0589353992753186e-2 5.197237388400598e-2 9.335842702194047e-2 1.0236898217883761e-1 0e0
/// 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 1.1134617525887985e-4 1.685563879327799e-3 1.0890236822613256e-2 3.593971402808193e-2 7.825268684067159e-2 9.70187210234942e-2 0e0
/// 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 3.4419923061348453e-4 4.05381058194703e-3 2.10564908490968e-2 6.218612252984003e-2 9.445607308437176e-2 0e0
/// 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 9.609032490541108e-4 1.0295385408099565e-2 4.56242880491308e-2 9.8251139498034e-2 0e0
/// 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 3.656397705090221e-3 3.337130404635729e-2 9.823157730002544e-2 0e0
/// 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 1.518498340844931e-2 1.0283847493104326e-1 0e0
/// 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 9.891625423032532e-2 0e0
/// 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 0e0 1e0
/// EOD
/// splot $data matrix with image t "" 
/// set output
/// ```
/// 
/// # Example: Replica exchange wang landau
/// * The same example as above, but with replica exchange wang landau
/// * [see explanaition of the model](#example-coin-flips)
/// ```
/// // feature is activated by default
/// #[cfg(feature="replica_exchange")]
/// {
/// use rand::SeedableRng;
/// use rand_pcg::Pcg64;
/// use sampling::{*, examples::coin_flips::*};
/// use std::{num::*, time::*};
/// use statrs::distribution::{Binomial, Discrete};
/// use std::fs::File;
/// use std::io::{BufWriter, Write};
///
/// let begin = Instant::now();
/// // length of coin flip sequence
/// let n = 20;
/// // how many intervals do we want?
/// let interval_count = 3;
///
/// // create histogram. The result of our `energy` (number of heads) can be anything between 0 and n
/// let hist = HistUsizeFast::new_inclusive(0, n).unwrap();
///
/// // now the overlapping histograms for sampling
/// // lets create 3 histograms. The parameter overlap here must be larger than 0. Normally, 2 should be good
/// let hist_list = hist.overlapping_partition(interval_count, 2).unwrap();
/// let rng = Pcg64::seed_from_u64(19756556678);
/// 
/// // create an ensemble
/// let ensemble = CoinFlipSequence::new(n, rng);
/// 
/// // create the replica exchange simulation builder. Note: There are different methods for this
/// let rewl_builder = RewlBuilder::from_ensemble(
///     ensemble,                           // the ensemble
///     hist_list,                          // the histograms, used as intervals for the Rewl
///     1,                                  // step size for the markov steps
///     NonZeroUsize::new(3000).unwrap(),   // sweep size, i.e., how many steps will be performed before a replica exchange will be tried
///     NonZeroUsize::new(4).unwrap(),      // How many random walkers should sample each interval (independently)?
///     0.0000025                           // Threshold for the simulation
/// ).unwrap();
///
/// // Note: You can now change the sweep size and the step sizes for the different 
/// // intervals independently.
/// // Use the `rewl_builder.step_sizes_mut()` and `rewl_builder.sweep_sizes.mut()` respecively
/// // The indice in the slices corresponds to the interval(index)
/// 
/// // uses greedy heuristik to find valid starting point.
/// // (fastest, if the ensembles are already at their respective vaild starting points)
/// // Note: there are different heuristiks. You have to try them out to see, which works best for your problem
/// let mut rewl = rewl_builder
///     .greedy_build(
///         |e| Some(e.head_count()) // energy function. It is a logical error to use a different energy function later on
///     );
/// 
/// // lets say, we want to limit our simulation to roughly 40 minutes at most
/// let start = Instant::now();
/// let seconds = 40 * 60; // seconds in 40 minutes
///         
/// // This is the heart pice - it performs the actual simulation
/// rewl.simulate_while(
///     |e| Some(e.head_count()), // energy function. has to be the same as used above
///     |_| start.elapsed().as_secs() < seconds // condition for continuation of simulation
/// );
/// // note, the above simulation might take slightly longer than 40 seconds, 
/// // because the condition is only checked after each sweep
///
/// // now lets get the result of the simulation:
/// // The logarithm (here base e, there is also a function for base 10) of the probability density (or density of states)
/// let glued = rewl.derivative_merged_log_prob_and_aligned()
///     .unwrap();
/// 
/// let ln_prob = glued.glued();
/// 
///
/// // For this example, we know the exact result. Lets calculate it to compare
/// let binomial = Binomial::new(0.5, n as u64).unwrap();
/// let ln_prob_true: Vec<_> = (0..=n)
///     .map(|k| binomial.ln_pmf(k as u64))
///     .collect();
///
/// let mut max_ln_difference = f64::NEG_INFINITY;
/// let mut max_difference = f64::NEG_INFINITY;
/// let mut frac_difference_max = f64::NEG_INFINITY;
/// let mut frac_difference_min = f64::INFINITY;
/// for (index, val) in ln_prob.into_iter().zip(ln_prob_true.into_iter()).enumerate()
/// {
///     println!("{} {} {}", index, val.0, val.1);
///     let val_simulation = val.0.exp();
///     let val_real = val.1.exp();
///     max_difference = f64::max((val_simulation - val_real).abs(), max_difference);
///     max_ln_difference = f64::max(max_ln_difference, (val.0-val.1).abs());
/// 
///     let frac = val_simulation / val_real;
///     frac_difference_max = frac_difference_max.max(frac);
///     frac_difference_min = frac_difference_min.min(frac);
///     
/// }
///
/// println!("max_ln_difference: {}", max_ln_difference);
/// println!("max absolute difference: {}", max_difference);
/// println!("max frac: {}", frac_difference_max);
/// println!("min frac: {}", frac_difference_min);
///
/// // at worst the simulated density overetimated the real result by under 1 %
/// assert!((frac_difference_max - 1.0).abs() < 0.01);
/// // and underestimated the result by under 1 %
/// assert!((frac_difference_min - 1.0).abs() < 0.01);
/// 
/// 
/// // Note: to get even better results, you can decrease the threshold 
/// // I used 2.5E-6. Often it is good to use between 1E-6 and 1E-8
/// // I used a larger threshold, since this is also a doc test and 
/// // should run under 5 minutes in Debug mode
/// // (on my machine it takes about 30 seconds in debug mode and under 2 seconds in release mode)
/// 
/// // if you want to see, how good the intervals align, you can do the following
/// 
/// let file = File::create("coin_flip_rewl.dat").unwrap();
/// let buf = BufWriter::new(file);
/// 
/// let glued = rewl.derivative_merged_log_prob_and_aligned().unwrap();
/// 
/// glued.write(buf).unwrap();
/// 
/// 
/// println!("Total time: {}", begin.elapsed().as_secs());
/// }
/// ```
/// 
/// To plot it, use gnuplot with
/// ```gp
/// set format y "e^{%.0f}"
/// set ylabel "probability"
/// set xlabel "Number of heads"
/// p "coin_flip_rewl.dat" u 1:2 t "merged", for[i=3:5] "" u 1:i t "interval ".(i-3)
/// ```
/// 
/// The resulting file is "coin_flip_rewl.dat"
/// ```dat
///#left_border right_border merged_ln interval0_ln interval1_ln …
///#bin log_merged log_interval0 …
///#log: BaseE
///0 -1.3871903700121702e1 -1.3875739115786455e1 NaN NaN
///1 -1.0870146579412317e1 -1.087398199507707e1 NaN NaN
///2 -8.618332301859919e0 -8.622167717524672e0 NaN NaN
///3 -6.82920349878786e0 -6.8330389144526125e0 NaN NaN
///4 -5.377758170163912e0 -5.381593585828665e0 -5.388874667753538e0 NaN
///5 -4.216803437381242e0 -4.220638853045995e0 -4.226280476088033e0 NaN
///6 -3.29812994129598e0 -3.301965356960733e0 -3.305497156868495e0 NaN
///7 -2.605482088178616e0 -2.6093175038433687e0 -2.6108809938156528e0 NaN
///8 -2.1197452928056864e0 -2.1235807084704392e0 -2.1240478848774105e0 -2.122542340935621e0
///9 -1.8310804880851892e0 -1.834915903749942e0 -1.834915903749942e0 -1.8347018659540995e0
///10 -1.735772920315168e0 -1.7376906281475444e0 -1.7396083359799208e0 -1.7396083359799208e0
///11 -1.831533663302804e0 -1.8331206687810608e0 -1.8363372589726454e0 -1.835369078967557e0
///12 -2.1181512793901733e0 -2.1199758856386497e0 -2.1267731198929996e0 -2.121986695054926e0
///13 -2.60403923032207e0 NaN -2.613084660078397e0 -2.607874645986823e0
///14 -3.2983258973492697e0 NaN -3.3051691503818974e0 -3.3021613130140226e0
///15 -4.216952145493605e0 NaN -4.21975952523182e0 -4.220787561158358e0
///16 -5.381221237813844e0 NaN -5.381876163859073e0 -5.385056653478597e0
///17 -6.829471030462838e0 NaN NaN -6.833306446127591e0
///18 -8.62372738543517e0 NaN NaN -8.627562801099923e0
///19 -1.0870037296186052e1 NaN NaN -1.0873872711850805e1
///20 -1.3867708133842694e1 NaN NaN -1.3871543549507447e1
/// ```
pub mod coin_flips;