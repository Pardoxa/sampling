pub mod coin_flips;


#[cfg(test)]
mod tests{
    use super::*;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;
    use coin_flips::*;
    use crate::*;
    use std::fs::File;
    use std::io::{BufWriter, Write};
    use statrs::distribution::{Binomial, Discrete};
    #[test]
    fn coins(){
        // Lets assume we do n Coinflips and want to measure the probability for the number of times,
        // this results in Head. This means, the number of times the coin flip returned head is the `energy`
        //
        // Of cause, for this example there is a analytic solution.


        // length of coin flip sequence
        let n = 20;
        let interval_count = 3;

        // create histogram. The result of our `energy` (number of heads) can be anything between 0 and n
        let hist = HistUsizeFast::new_inclusive(0, n).unwrap();

        // now the overlapping histograms for sampling
        // lets create 3 histograms. Overlap should be larger than 0. Normally, 1 is sufficient
        let hist_list = hist.overlapping_partition(interval_count, 1).unwrap();

        // create rng to seed all other rngs
        let mut rng = Pcg64::seed_from_u64(834628956578);

        // now create ensembles (could be combined with wl creation)
        // note: You could also create one ensemble and clone it instead of creating different ones
        let ensembles: Vec<_> = (0..interval_count).map(|_| {
            CoinFlipSequence::new(
                n,
                Pcg64::from_rng(&mut rng).unwrap()
            )
        }).collect();

        // Now the Wang Landau simulation. First create the struct (here as Vector, since we have 3 intervals)
        let mut wl_list: Vec<_> = ensembles.into_iter()
            .zip(hist_list.into_iter())
            .map(|(ensemble, histogram)| {
                WangLandau1T::new(
                    0.00001, // arbitrary threshold, you have to try what is good for your model
                    ensemble,
                    Pcg64::from_rng(&mut rng).unwrap(),
                    1,  // stepsize 1 is sufficient for this problem
                    histogram,
                    100
                ).unwrap()
            }).collect();

        // Now we have to initialize the wl with a valid state
        // as the simulation has to start in the interval one wants to measure.
        // Since the energy landscape is quite simple, here a greedy approach is good enough.
        
        wl_list.iter_mut()
            .for_each(|wl|{
                wl.init_greedy_heuristic(
                    |coin_seq| Some(coin_seq.head_count()), // Our ensemble always retuns a valid state - thus I use Some here
                    Some(10_000) // if no valid state is found after 10_000 this returns an Err. If you do not want a step limit, you can use None here
                ).expect("Unable to find valid state within 10_000 steps!");
            });

        // Now our ensemble is initialized. Time for the Wang Landau Simulation. You can do that in different ways.
        // I will show this by doing it differently for our three intervals

        // First, the simplest one. Just simulate until it is converged.
        wl_list[0].wang_landau_convergence(
            |coin_seq| Some(coin_seq.head_count())
        );

        // Secondly, I only have a limited amount of time.
        // Lets say, I have 1 minute at most.
        let start_time = std::time::Instant::now();
        wl_list[1].wang_landau_while(
            |coin_seq| Some(coin_seq.head_count()),
            |_| start_time.elapsed().as_secs() <= 60
        );

        // Now, lets see if our last two simulations did indeed finish:
        assert!(wl_list[1].is_finished());

        // Or lets say, I want to limit the number of steps to 100_000
        wl_list[2].wang_landau_while(
            |coin_seq| Some(coin_seq.head_count()),
            |state| state.step_counter() <= 100_000 
        );

        // This simulation did not finish
        assert!(!wl_list[2].is_finished());

        // If it did not finish, you could, e.g., store the state using serde.
        // I recommend the crate `bincode` for storing
        // Than you could continue the simulation later on.
        
        // lets resume the simulation for now
        wl_list[2].wang_landau_convergence(
            |coin_seq| Some(coin_seq.head_count())
        );
        // it finished
        assert!(wl_list[2].is_finished());

        // Since our simulations did all finish, lets see what our distribution looks like
        // Lets glue them together. We use our original histogram for that.
        let glued = glue_wl(
            &wl_list,
            &hist
        ).expect("Unable to glue results. Look at error message");

        // now, lets print our result
        glued.write(std::io::stdout()).unwrap();

        // or store it into a file
        let file = File::create("coin_flip_log_density.dat").unwrap();
        let buf = BufWriter::new(file);
        glued.write(buf).unwrap();
        
        // now, lets check if our results are actually any good.
        let log10_prob = glued.glued_log10_probability;
        
        // lets compare that to the analytical result
        // Since the library I am going to use lets me directly calculate the natural
        // logaritm of the probability, I first convert the base of our own results:
        let ln_prob: Vec<_> = log10_prob.iter().map(|&val| val / std::f64::consts::LOG10_E).collect();
        
        // Then create the `true` results:
        let binomial = Binomial::new(0.5, n as u64).unwrap();

        let ln_prob_true: Vec<_> = (0..=n)
            .map(|k| binomial.ln_pmf(k as u64))
            .collect();

        // lets write that in a file, so we can use gnuplot to plot the result
        // lets also calculate the maximum difference between the two solutions
        let mut max_ln_dif = std::f64::NEG_INFINITY;
        let mut max_dif = std::f64::NEG_INFINITY;

        let comp_file = File::create("coin_flip_compare.dat").unwrap();
        let mut buf = BufWriter::new(comp_file);
        
        writeln!(buf, "#head_count Numeric_ln_prob Analytic_ln_prob ln_dif dif").unwrap();
        for (index, (numeric, analytic)) in ln_prob.iter().zip(ln_prob_true.iter()).enumerate()
        {
            let ln_dif = numeric - analytic;
            max_ln_dif = ln_dif.abs().max(max_ln_dif);
            let dif = 10_f64.powf(*numeric) - 10_f64.powf(*analytic);
            max_dif = dif.abs().max(max_dif);
            writeln!(buf, "{} {} {} {} {}", index, numeric, analytic, ln_dif, dif).unwrap();
        }

        println!("Max_ln_dif = {}", max_ln_dif);
        println!("Max_dif = {}", max_dif);

        // in this case, the max difference of the natural logarithms of the probabilities is smaller than 0.03
        assert!(max_ln_dif < 0.03);
        // and the max absolut difference is smaler than 0.0002
        assert!(max_dif < 0.0002);

        // But we can do better. Lets refine the results with entropic sampling
        // first, convert the wl simulations in entropic sampling simulations
        let mut entropic_list: Vec<_> = wl_list
            .into_iter()
            .map(|wl| EntropicSampling::from_wl(wl).unwrap())
            .collect();


        // Now, while doing that, lets also create a heatmap.
        // We want to see, how the number of heads correlates to the maximum number of heads in a row.

        // In this case, the heatmap is symetric and we already have a histogram of correct sice
        let mut heatmap = HeatmapU::new(
            hist.clone(),
            hist.clone()
        );

        entropic_list.iter_mut()
            .for_each(|entr|{
                entr.entropic_sampling(
                    |coin_seq| Some(coin_seq.head_count()),
                    |state| {
                        let head_count = *state.energy();
                        let heads_in_row = state.ensemble().max_heads_in_a_row();
                        heatmap.count(head_count, heads_in_row)
                            .expect("Value outside heatmap?");
                    }
                )
            });
        
        // Now, lets see our refined results:
        let glued = glue_entropic(
            &entropic_list,
            &hist
        ).expect("Unable to glue results. Look at error message");

        // lets store our result
        let file = File::create("coin_flip_log_density_entropic.dat").unwrap();
        let buf = BufWriter::new(file);
        glued.write(buf).unwrap();

        // now, lets compare with the analytical results again
        // Again, calculate to base e
        let ln_prob: Vec<_> = glued.glued_log10_probability
            .iter()
            .map(|&val| val / std::f64::consts::LOG10_E)
            .collect();

        
        // lets write that in a file, so we can use gnuplot to plot the result
        // lets also calculate the maximum difference between the two solutions
        let mut max_ln_dif = std::f64::NEG_INFINITY;
        let mut max_dif = std::f64::NEG_INFINITY;

        let comp_file = File::create("coin_flip_compare_entr.dat").unwrap();
        let mut buf = BufWriter::new(comp_file);
        
        writeln!(buf, "#head_count Numeric_ln_prob Analytic_ln_prob ln_dif dif").unwrap();
        for (index, (numeric, analytic)) in ln_prob.iter().zip(ln_prob_true.iter()).enumerate()
        {
            let ln_dif = numeric - analytic;
            max_ln_dif = ln_dif.abs().max(max_ln_dif);
            let dif = 10_f64.powf(*numeric) - 10_f64.powf(*analytic);
            max_dif = dif.abs().max(max_dif);
            writeln!(buf, "{} {} {} {} {}", index, numeric, analytic, ln_dif, dif).unwrap();
        }

        println!("Max_ln_dif = {}", max_ln_dif);
        println!("Max_dif = {}", max_dif);

        // in this case, the max difference of the natural logarithms of the probabilities is smaller than 0.03
        assert!(max_ln_dif < 0.026);
        // and the max absolut difference is smaller than 0.0002
        assert!(max_dif < 0.0002);

        // That would be the final result for our probability density than. As you can see, it is very very 
        // close to the analytical result.

        // Now, lets see, how our heatmap looks:
        let mut settings = GnuplotSettings::new();
        settings.x_label("#Heads")
            .y_label("Max heads in row");

        // lets normalize coloumwise 
        let heatmap = heatmap.heatmap_normalized_columns();

        // now create gnuplot file
        let file = File::create("coin_heatmap.gp").unwrap();
        let buf = BufWriter::new(file);
        heatmap.gnuplot(
            buf,
            "heatmap_coin_flips",
            settings
        ).unwrap();

        // now you can use gnuplot to plot the heatmap
    }
}