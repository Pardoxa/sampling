use{
    crate::{*, traits::*},
    rand::Rng,
    num_traits::{Bounded, ops::wrapping::*, identities::*},
    std::{marker::PhantomData, io::Write, num::*}
};

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

/// # The 1/t Wang Landau approach comes from this paper
/// > R. E. Belardinelli and V. D. Pereyra,
/// > “Fast algorithm to calculate density of states,”
/// > Phys.&nbsp;Rev.&nbsp;E&nbsp;**75**: 046701 (2007), DOI&nbsp;[10.1103/PhysRevE.75.046701](https://doi.org/10.1103/PhysRevE.75.046701)
/// 
/// * The original Wang Landau algorithim comes from this paper
/// > F. Wang and D. P. Landau,
/// > “Efficient, multiple-range random walk algorithm to calculate the density of states,” 
/// > Phys.&nbsp;Rev.&nbsp;Lett.&nbsp;**86**, 2050–2053 (2001), DOI&nbsp;[10.1103/PhysRevLett.86.2050](https://doi.org/10.1103/PhysRevLett.86.2050)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct WangLandau1T<Hist, Rng, Ensemble, S, Res, Energy>{
    pub(crate) ensemble: Ensemble,
    pub(crate) rng: Rng,
    pub(crate) marker_res: PhantomData<Res>,
    pub(crate) steps: Vec<S>,
    mode: WangLandauMode,
    pub(crate) log_density: Vec<f64>,
    pub(crate) log_f: f64,
    pub(crate) log_f_threshold: f64,
    pub(crate) step_size: usize,
    step_count: usize,
    accepted_steps_total: usize,
    recected_steps_total: usize,
    accepted_steps_current: usize,
    recected_steps_current: usize,
    pub(crate) old_bin: usize,
    pub(crate) hist: Hist,
    pub(crate) old_energy: Option<Energy>,
    check_refine_every: usize,
}

impl<Hist, Rng, Ensemble, S, Res, Energy>WangLandau1T<Hist, Rng, Ensemble, S, Res, Energy>
{
    /// Returns internal ensemble, histogram and Rng
    pub fn into_inner(self) -> (Ensemble, Hist, Rng)
    {
        (self.ensemble, self.hist, self.rng)
    }
}


impl<Hist, R, E, S, Res, Energy> WangLandau 
    for WangLandau1T<Hist, R, E, S, Res, Energy>
{
    #[inline(always)]
    fn log_f(&self) -> f64
    {
        self.log_f
    }

    #[inline(always)]
    fn log_f_threshold(&self) -> f64
    {
        self.log_f_threshold
    }

    fn set_log_f_threshold(&mut self, log_f_threshold: f64) -> Result<f64, WangLandauErrors>
    {
        if !log_f_threshold.is_finite() || log_f_threshold.is_sign_negative() {
            return Err(WangLandauErrors::InvalidLogFThreshold);
        }
        let old_threshold = self.log_f_threshold;
        self.log_f_threshold = log_f_threshold;
        Ok(old_threshold)
    }

    #[inline(always)]
    fn log_density(&self) -> &Vec<f64>
    {
        &self.log_density
    }

    fn write_log<W: Write>(&self, mut writer: W) -> Result<(), std::io::Error> {
        writeln!(writer,
            "#Acceptance prob_total: {}\n#Acceptance prob current: {}\n#total_steps: {}\n#log_f: {:e}\n#Current_mode {:?}",
            self.fraction_accepted_total(),
            self.fraction_accepted_current(),
            self.step_counter(),
            self.log_f(),
            self.mode
        )?;
        writeln!(
            writer,
            "#total_steps_accepted: {}\n#total_steps_rejected: {}\n#current_accepted_steps: {}\n#current_rejected_steps: {}",
            self.accepted_steps_total,
            self.recected_steps_total,
            self.accepted_steps_current,
            self.recected_steps_current
        )
    }

    #[inline(always)]
    fn mode(&self) -> WangLandauMode
    {
        self.mode
    }

    #[inline(always)]
    fn step_counter(&self) -> usize
    {
        self.step_count
    }

    #[inline(always)]
    fn total_steps_rejected(&self) -> usize {
        self.recected_steps_total
    }

    #[inline(always)]
    fn total_steps_accepted(&self) -> usize {
        self.accepted_steps_total
    }
}

impl<Hist, R, E, S, Res, Energy> 
    WangLandau1T<Hist, R, E, S, Res, Energy>
where 
    Hist: Histogram + HistogramVal<Energy>
{
    /// # Check if `self` is initialized
    /// * if this returns true, you can begin the WangLandau simulation
    /// * otherwise call one of the `self.init*` methods
    pub fn is_initialized(&self) -> bool
    {
        match &self.old_energy{
            None => false,
            Some(e) => {
                self.hist.is_inside(e)
            }
        }
    }
}

/// Possible errors when setting initial guess
#[derive(Clone, Copy, Debug)]
pub enum SetInitialError{
    /// # Dimensions do not match! 
    /// The length of the initial guess and the amount of bins have to be the same 
    DimensionError,
    /// All values inside the initial guess have to be finite
    NonFiniteEncountered,
    /// log_f has to fullfill 0.0 < log_f < 10.0
    InvalidLogF
}


impl <Hist, R, E, S, Res, Energy> WangLandauEnsemble<E> 
    for WangLandau1T<Hist, R, E, S, Res, Energy>
{
    #[inline(always)]
    fn ensemble(&self) -> &E {
        &self.ensemble
    }

    #[inline(always)]
    unsafe fn ensemble_mut(&mut self) -> &mut E {
        &mut self.ensemble
    }
}

impl <Hist, R, E, S, Res, Energy> WangLandauEnergy<Energy> 
    for WangLandau1T<Hist, R, E, S, Res, Energy>
{
    #[inline(always)]
    fn energy(&self) -> Option<&Energy> {
        self.old_energy.as_ref()
    }
}

impl <Hist, R, E, S, Res, Energy> WangLandauHist<Hist> 
    for WangLandau1T<Hist, R, E, S, Res, Energy>
{
    #[inline(always)]
    fn hist(&self) -> &Hist {
        &self.hist   
    }
}

impl<Hist, R, E, S, Res, Energy> 
    WangLandau1T<Hist, R, E, S, Res, Energy>
{
    /// # Acceptance rate
    /// Fraction of performed wang landau steps, that were accepted
    fn fraction_accepted_total(&self) -> f64
    {
        let sum = self.accepted_steps_total + self.recected_steps_total;
        self.accepted_steps_total as f64 / sum as f64
    }

    /// # Acceptance rate since last Refinement
    /// Fraction of performed wang landau steps since 
    /// the last time, the factor f was refined, that were accepted
    fn fraction_accepted_current(&self) -> f64
    {
        let total = self.accepted_steps_current + self.recected_steps_current;
        if total == 0 {
            f64::NAN
        } else {
            self.accepted_steps_current as f64 / total as f64
        }
    }

    /// # Set the initial guess for the non-normalized probability estimate
    /// * `new_guess` your new guess for the probability estimate. Its length has to equal the number of bins of the internal histogram
    /// which is the same as the length of the old estimate which you can get by calling [log_density](Self::log_density). All contained values have 
    /// to be finite
    /// * `new_log_f`: Which log_f to start at? 0.0 < log_f <= 10.0 has to be true. 
    /// If you don't know what's best I recommand starting with log_f=1.0, the better your probability estimate is, the smaller this value can be
    /// # Note
    /// This will reset the calculation. Meaning you will have to call one of the initializing functions like `init_greedy_heuristic`again 
    /// and all internal counters are reset to 0
    pub fn set_initial_probability_guess(mut self, new_guess: Vec<f64>, new_log_f: f64) -> Result<Self, SetInitialError>
    where Hist: Histogram
    {
        if 0.0 >= new_log_f || new_log_f > 10.0 {
            Err(SetInitialError::InvalidLogF)
        }
        else if new_guess.len() != self.log_density.len()
        { 
            Err(SetInitialError::DimensionError)
        } else if new_guess.iter().any(|val| !val.is_finite())
        {
            Err(SetInitialError::NonFiniteEncountered)
        } else {
            self.log_density = new_guess;
            self.log_f = new_log_f;
            self.step_count = 0;
            self.accepted_steps_current = 0;
            self.accepted_steps_total = 0;
            self.recected_steps_current = 0;
            self.recected_steps_total = 0;
            self.mode = WangLandauMode::RefineOriginal;
            self.hist.reset();
            self.old_energy = None;
            self.old_bin = usize::MAX;
            Ok(self)
        }
    }
}


impl<Hist, R, E, S, Res, Energy> 
    WangLandau1T<Hist, R, E, S, Res, Energy>
where 
    R: Rng,
    E: MarkovChain<S, Res>,
    Energy: Clone,
    Hist: Histogram + HistogramVal<Energy>
{
    /// # Create a new WangLandau simulation
    /// **IMPORTANT** You have to call one of the `init*` functions, 
    /// to create a valid state, before you can start the simulation
    /// ## Parameter
    /// * `log_f_threshold`: how small should the ln(f) (see paper) become
    /// until the simulation is finished?
    /// * `ensemble`: The ensemble to explore. 
    /// Current state of ensemble will be used as inital condition for the `init*` functions
    /// * `step_size`: The markov steps will be performed with this step size, e.g., 
    /// `ensemble.m_steps(step_size)`
    /// * `histogram`: Provides the binning. You can either use one of the already implemented
    /// histograms, like `HistU32Fast`, `HistU32`, `HistF64` etc. or implement your own by 
    /// implementing the traits `Histogram + HistogramVal<Energy>` yourself
    /// * `check_refine_every`: how often to check, if every bin in the histogram was hit.
    /// Needs to be at least 1. Good values depend on the problem at hand, but if you are 
    /// unsure, you can start with a value like 1000 
    pub fn new(
        log_f_threshold: f64,
        ensemble: E,
        rng: R,
        step_size: usize,
        histogram: Hist,
        check_refine_every: usize
    )-> Result<Self, WangLandauErrors>
    {
        if !log_f_threshold.is_finite() || log_f_threshold.is_sign_negative() 
        {
            return Err(WangLandauErrors::InvalidLogFThreshold);
        }
        else if check_refine_every == 0 {
            return Err(WangLandauErrors::CheckRefineEvery0)
        }
        let log_density = vec![0.0; histogram.bin_count()];
        let steps = Vec::with_capacity(step_size);

        Ok(
            Self{
                ensemble,
                step_count: 0,
                step_size,
                hist: histogram,
                rng,
                marker_res: PhantomData::<Res>,
                log_f: 1.0,
                log_density,
                log_f_threshold,
                mode: WangLandauMode::RefineOriginal,
                recected_steps_current: 0,
                recected_steps_total: 0,
                accepted_steps_current: 0,
                accepted_steps_total: 0,
                old_bin: usize::MAX,
                old_energy: None,
                check_refine_every,
                steps,
            }
        )
    }

    fn init<F>(
        &mut self,
        energy_fn: F,
        step_limit: Option<u64>
    ) -> Result<(), WangLandauErrors>
        where F: Fn(&mut E) -> Option<Energy>
    {
        self.old_energy = energy_fn(&mut self.ensemble);
        if self.old_energy.is_some(){
            return Ok(());
        }

        match step_limit {
            None => {
                loop {
                    self.ensemble.m_steps_quiet(self.step_size);
                    self.old_energy = energy_fn(&mut self.ensemble);
        
                    if self.old_energy.is_some(){
                        self.count_accepted();
                        return Ok(());
                    }
                    self.count_rejected();
                }
            },
            Some(limit) => {
                for _ in 0..limit {
                    self.ensemble.m_steps_quiet(self.step_size);
                    self.old_energy = energy_fn(&mut self.ensemble);
        
                    if self.old_energy.is_some(){
                        self.count_accepted();
                        return Ok(());
                    }
                    self.count_rejected();
                }
                Err(WangLandauErrors::InitFailed)
            }
        }
    }

    fn greedy_helper<F, H, J>(
        &mut self,
        old_distance: &mut J,
        energy_fn: F,
        distance_fn: H,
    )   where F: Fn(&mut E) -> Option<Energy> + Copy,
            H: Fn(&Hist, &Energy) -> J,
            J: PartialOrd
    {
        self.ensemble
            .m_steps(self.step_size, &mut self.steps);

        
        if let Some(energy) = energy_fn(&mut self.ensemble) {
            let distance = distance_fn(&self.hist, &energy);
            if distance <= *old_distance {
                self.old_energy = Some(energy);
                *old_distance = distance;
                self.count_accepted();
                
                return;
            }
        }

        
        self.count_rejected();
        self.ensemble
            .undo_steps_quiet(&self.steps);
        
    }

    /// # Find a valid starting Point
    /// * if the ensemble is already at a valid starting point,
    /// the ensemble is left unchanged (as long as your energy calculation does not change the ensemble)
    /// * Uses a greedy heuristik. Performs markov steps. If that brought us closer to the target interval,
    /// the step is accepted. Otherwise it is rejected
    /// # Parameter
    /// * `step_limit`: Some(val) -> val is max number of steps tried, if no valid state is found, it will return an Error. None -> will loop until either 
    /// a valid state is found or forever
    /// * `energy_fn` function calculating `Some(energy)` of the system
    /// or rather the Parameter of which you wish to obtain the probability distribution.
    ///  Has to be the same function as used for the wang landau simulation later.
    /// If there are any states, for which the calculation is invalid, `None` should be returned
    /// * steps resulting in ensembles for which `energy_fn(&mut ensemble)` is `None`
    /// will always be rejected 
    pub fn init_greedy_heuristic<F>(
        &mut self,
        energy_fn: F,
        step_limit: Option<u64>,
    ) -> Result<(), WangLandauErrors>
    where F: Fn(&mut E) -> Option<Energy>,
    {
        self.init(&energy_fn, step_limit)?;
        let mut old_distance = self.hist
            .distance(self.old_energy_ref());
        let mut step_count = 0;
        while old_distance != 0.0 {
            self.greedy_helper(
                &mut old_distance,
                &energy_fn,
                |hist, energy| {
                    hist.distance(energy)
                }
            );
            if let Some(limit) = step_limit {
                if limit == step_count{
                    return Err(WangLandauErrors::InitFailed);
                }
                step_count += 1;
            }
        }
        self.end_init();
        Ok(())
    }

    /// # Find a valid starting Point
    /// * if the ensemble is already at a valid starting point,
    /// the ensemble is left unchanged (as long as your energy calculation does not change the ensemble)
    /// * Uses overlapping intervals. Accepts a step, if the resulting ensemble is in the same interval as before,
    /// or it is in an interval closer to the target interval
    /// * Take a look at the [`HistogramIntervalDistance` trait](`crate::HistogramIntervalDistance`)
    /// # Parameter
    /// * `step_limit`: Some(val) -> val is max number of steps tried, if no valid state is found, it will return an Error. None -> will loop until either 
    /// a valid state is found or forever
    /// * `energy_fn` function calculating `Some(energy)` of the system
    /// or rather the Parameter of which you wish to obtain the probability distribution.
    ///  Has to be the same function as used for the wang landau simulation later.
    /// If there are any states, for which the calculation is invalid, `None` should be returned
    /// * steps resulting in ensembles for which `energy_fn(&mut ensemble)` is `None`
    /// will always be rejected 
    pub fn init_interval_heuristik<F>(
        &mut self,
        overlap: NonZeroUsize,
        energy_fn: F,
        step_limit: Option<u64>,
    ) -> Result<(), WangLandauErrors>
    where F: Fn(&mut E) -> Option<Energy>,
        Hist: HistogramIntervalDistance<Energy>
    {
        self.init(&energy_fn, step_limit)?;
        let mut old_dist = self.hist
            .interval_distance_overlap(
                self.old_energy_ref(),
                overlap
            );
        
        let dist = |h: &Hist, val: &Energy| h.interval_distance_overlap(val, overlap);
        let mut step_count = 0;
        while old_dist != 0 {
            self.greedy_helper(
                &mut old_dist,
                &energy_fn,
                dist
            );
            if let Some(limit) = step_limit {
                if limit == step_count{
                    return Err(WangLandauErrors::InitFailed);
                }
                step_count += 1;
            }
        }
        self.end_init();
        Ok(())
    }

    /// # Find a valid starting Point
    /// * if the ensemble is already at a valid starting point,
    /// the ensemble is left unchanged (as long as your energy calculation does not change the ensemble)
    /// * `overlap` - see [`HistogramIntervalDistance` trait](`crate::HistogramIntervalDistance`)
    /// Should be greater than 0 and smaller than the number of bins in your histogram. E.g. `overlap = 3` if you have 200 bins
    /// * `mid` - should be something like `128u8`, `0i8` or `0i16`. It is very unlikely that using a type with more than 16 bit makes sense for mid
    /// * `step_limit`: Some(val) -> val is max number of steps tried, if no valid state is found, it will return an Error. None -> will loop until either 
    /// a valid state is found or forever
    /// * alternates between greedy and interval heuristik everytime a wrapping counter passes `mid` or `U::min_value()`
    /// * I recommend using this heuristik, if you do not know which one to use
    /// # Parameter
     /// * `energy_fn` function calculating `Some(energy)` of the system
    /// or rather the Parameter of which you wish to obtain the probability distribution.
    ///  Has to be the same function as used for the wang landau simulation later.
    /// If there are any states, for which the calculation is invalid, `None` should be returned
    /// * steps resulting in ensembles for which `energy_fn(&mut ensemble)` is `None`
    /// will always be rejected 
    pub fn init_mixed_heuristik<F, U>
    (
        &mut self,
        overlap: NonZeroUsize,
        mid: U,
        energy_fn: F,
        step_limit: Option<u64>
    ) -> Result<(), WangLandauErrors>
    where F: Fn(&mut E) -> Option<Energy>,
        Hist: HistogramIntervalDistance<Energy>,
        U: One + Bounded + WrappingAdd + Eq + PartialOrd
    {
        self.init(&energy_fn, step_limit)?;
        if self.hist
            .is_inside(
                self.old_energy_ref()
            )
        {
            self.end_init();
            return Ok(());
        }    
        
        let mut old_dist = f64::INFINITY;
        let mut old_dist_interval = usize::MAX;
        let mut counter: U = U::min_value();
        let min_val = U::min_value();
        let one = U::one();
        let dist_interval = |h: &Hist, val: &Energy| h.interval_distance_overlap(val, overlap);
        let mut step_count = 0;
        loop {
            if counter == min_val {
                let current_energy = self.old_energy_ref();
                old_dist = self.hist.distance(current_energy);
            }else if counter == mid {
                let current_energy = self.old_energy_ref();
                old_dist_interval = dist_interval(&self.hist, current_energy);
            }
            if counter < mid {
                self.greedy_helper(
                    &mut old_dist,
                    &energy_fn,
                    |hist, val| {
                        hist.distance(val)
                    }
                );
                if old_dist == 0.0 {
                    break;
                }
            } else {
                self.greedy_helper(
                    &mut old_dist_interval,
                    &energy_fn,
                    dist_interval,
                );
                if old_dist_interval == 0 {
                    break;
                }
            }
            counter = counter.wrapping_add(&one);
            if let Some(limit) = step_limit {
                if limit == step_count{
                    return Err(WangLandauErrors::InitFailed);
                }
                step_count += 1;
            }
        }
        self.end_init();
        Ok(())
    }

    fn end_init(&mut self)
    {
        self.old_bin = self.hist
            .get_bin_index( 
                self.old_energy_ref()
            ).expect("Error in heuristic - old bin invalid");
    }

    fn old_energy_clone(&self) -> Energy {
        self.old_energy_ref()
            .clone()
    }

    fn old_energy_ref(&self) -> &Energy {
        self.old_energy
            .as_ref()
            .unwrap()
    }


    fn count_accepted(&mut self){
        self.ensemble.steps_accepted(&self.steps);
        self.accepted_steps_current += 1;
        self.accepted_steps_total += 1;
    }

    fn count_rejected(&mut self){
        self.ensemble.steps_rejected(&self.steps);
        self.recected_steps_current += 1;
        self.recected_steps_total += 1;
    }


    fn check_refine(&mut self)
    {
        match self.mode{
            WangLandauMode::Refine1T => {
                self.log_f = self.log_f_1_t();
            },
            WangLandauMode::RefineOriginal => {
                if self.step_count % self.check_refine_every == 0 
                    && !self.hist.any_bin_zero() 
                {
                    self.recected_steps_current = 0;
                    self.accepted_steps_current = 0;
                    let ref_1_t = self.log_f_1_t();
                    self.log_f *= 0.5;
                    if self.log_f < ref_1_t {
                        self.log_f = ref_1_t;
                        self.mode = WangLandauMode::Refine1T;
                    }
                    self.hist.reset();
                }
            }
        }
    }


    fn wl_step_helper(&mut self, energy: Option<Energy>)
    {
        let current_energy = match energy 
        {
            Some(energy) => energy,
            None => {
                self.count_rejected();
                self.hist.count_index(self.old_bin).unwrap();
                self.log_density[self.old_bin] += self.log_f;
                self.ensemble.undo_steps_quiet(&self.steps);
                return;
            }
        };
        
        match self.hist.get_bin_index(&current_energy)
        {
            Ok(current_bin) => {
                let accept_prob = self.metropolis_acception_prob( current_bin);

                if self.rng.gen::<f64>() > accept_prob {
                    // reject step
                    self.count_rejected();
                    self.ensemble.undo_steps_quiet(&self.steps);
                } else {
                    // accept step
                    self.count_accepted();
                    
                    self.old_energy = Some(current_energy);
                    self.old_bin = current_bin;
                }
            },
            _  => {
                // invalid step -> reject
                self.count_rejected();
                self.ensemble.undo_steps_quiet(&self.steps);
            }
        };
        
        self.hist.count_index(self.old_bin).unwrap();
        self.log_density[self.old_bin] += self.log_f;
    }

    /// # Wang Landau Step
    /// * performs a single Wang Landau step
    /// # Parameter
    /// * `energy_fn` function calculating `Some(energy)` of the system
    /// or rather the Parameter of which you wish to obtain the probability distribution.
    /// If there are any states, for which the calculation is invalid, `None` should be returned
    /// * steps resulting in ensembles for which `energy_fn(&mut ensemble)` is `None`
    /// will always be rejected 
    /// # Important
    /// * You have to call one of the `self.init*` functions before calling this one - 
    /// **will panic otherwise**
    pub fn wang_landau_step<F>(
        &mut self,
        energy_fn: F,
    )where F: Fn(&E) -> Option<Energy>
    {
        unsafe {
            self.wang_landau_step_unsafe(|e| energy_fn(e))
        }
    }

    /// # Wang Landau Step
    /// * if possible, use `self.wang_landau_step()` instead - it is safer
    /// * unsafe, because you have to make sure, that the `energy_fn` function 
    /// does not change the state of the ensemble in such a way, that the result of
    /// `energy_fn` changes when called again. Maybe do cleanup at the beginning of the energy function?
    /// * performs a single Wang Landau step
    /// # Parameter
    /// * `energy_fn` function calculating `Some(energy)` of the system
    /// or rather the Parameter of which you wish to obtain the probability distribution.
    /// If there are any states, for which the calculation is invalid, `None` should be returned
    /// * steps resulting in ensembles for which `energy_fn(&mut ensemble)` is `None`
    /// will always be rejected 
    /// # Important
    /// * You have to call one of the `self.init*` functions before calling this one - 
    /// **will panic otherwise**
    pub unsafe fn wang_landau_step_unsafe<F>(
        &mut self,
        mut energy_fn: F,
    )where F: FnMut(&mut E) -> Option<Energy>
    {
        debug_assert!(
            self.old_energy.is_some(),
            "Error - self.old_energy invalid - Did you forget to call one of the `self.init*` members for initialization?"
        );

        self.step_count += 1;


        self.ensemble.m_steps(self.step_size, &mut self.steps);
        
        self.check_refine();
        let current_energy = energy_fn(&mut self.ensemble);
        
        self.wl_step_helper(current_energy);
    }

    /// # Wang Landau Step
    /// * performs a single Wang Landau step
    /// # Parameter
    /// * `energy_fn` function calculating the energy of the system **on the fly**
    /// * **steps resulting in invalid ensembles are not allowed!**
    /// # Important
    /// * You have to call one of the `self.init*` functions before calling this one - 
    /// **will panic otherwise**
    pub fn wang_landau_step_acc<F>(
        &mut self,
        energy_fn: F,
    )
    where F: FnMut(&E, &S, &mut Energy)
    {
        debug_assert!(
            self.old_energy.is_some(),
            "Error - self.old_energy invalid - Did you forget to call one of the `self.init*` members for initialization?"
        );

        self.step_count += 1;

        let mut new_energy = self.old_energy_clone();

        self.ensemble
            .m_steps_acc(
                self.step_size,
                &mut self.steps,
                &mut new_energy,
                energy_fn
            );
        
        self.check_refine();
        
        self.wl_step_helper(Some(new_energy));

    }

    /// # Wang Landau
    /// * perform Wang Landau simulation
    /// * calls `self.wang_landau_step(energy_fn, valid_ensemble)` until `self.is_finished()` 
    pub fn wang_landau_convergence<F>(
        &mut self,
        energy_fn: F,
    )where F: Fn(&E) -> Option<Energy>,
    {
        while !self.is_finished() {
            self.wang_landau_step(&energy_fn);
        }
    }

    /// # Wang Landau - efficient energy calculation
    /// * perform Wang Landau simulation
    /// * calls `self.wang_landau_step_acc(energy_fn, valid_ensemble)` until `self.is_finished()` 
    pub fn wang_landau_convergence_acc<F>(
        &mut self,
        mut energy_fn: F,
    )where F: FnMut(&E, &S, &mut Energy)
    {
        while !self.is_finished() {
            self.wang_landau_step_acc(&mut energy_fn);
        }
    }

    /// # Wang Landau
    /// * if possible, use `self.wang_landau_convergence()` instead - it is safer
    /// * You have mutable access to your ensemble, which is why this function is unsafe. 
    /// If you do anything, which changes the future outcome of the energy function, the results will be wrong!
    /// * perform Wang Landau simulation
    /// * calls `self.wang_landau_step_unsafe(energy_fn, valid_ensemble)` until `self.is_finished()` 
    pub unsafe fn wang_landau_convergence_unsafe<F>(
        &mut self,
        mut energy_fn: F,
    )where F: FnMut(&mut E) -> Option<Energy>,
    {
        while !self.is_finished() {
            self.wang_landau_step_unsafe(&mut energy_fn);
        }
    }

    /// # Wang Landau
    /// * perform Wang Landau simulation
    /// * calls `self.wang_landau_step(energy_fn)` until `self.is_finished()` 
    /// or `condition(&self)` is false
    pub fn wang_landau_while<F, W>(
        &mut self,
        energy_fn: F,
        mut condition: W
    ) where F: Fn(&E) -> Option<Energy>,
        W: FnMut(&Self) -> bool,
    {
        while !self.is_finished() && condition(self) {
            self.wang_landau_step(&energy_fn);
        }
    }

    /// # Wang Landau
    /// * perform Wang Landau simulation
    /// * calls `self.wang_landau_step(energy_fn)` until `self.is_finished()` 
    /// or `condition(&self)` is false
    pub fn wang_landau_while_acc<F, W>(
        &mut self,
        mut energy_fn: F,
        mut condition: W
    ) where F: FnMut(&E, &S, &mut Energy),
        W: FnMut(&Self) -> bool,
    {
        while !self.is_finished() && condition(self) {
            self.wang_landau_step_acc(&mut energy_fn);
        }
    }

    /// # Wang Landau
    /// * if possible, use `self.wang_landau_while()` instead - it is safer
    /// * You have mutable access to your ensemble, which is why this function is unsafe. 
    /// If you do anything, which changes the future outcome of the energy function, the results will be wrong!
    /// * perform Wang Landau simulation
    /// * calls `self.wang_landau_step(energy_fn)` until `self.is_finished()` 
    /// or `condition(&self)` is false
    pub unsafe fn wang_landau_while_unsafe<F, W>(
        &mut self,
        mut energy_fn: F,
        mut condition: W
    ) where F: FnMut(&mut E) -> Option<Energy>,
        W: FnMut(&Self) -> bool,
    {
        while !self.is_finished() && condition(self) {
            self.wang_landau_step_unsafe(&mut energy_fn);
        }
    }


    /// **panics** if index is invalid
    #[inline(always)]
    fn metropolis_acception_prob(&self, new_bin: usize) -> f64
    {
        
        (self.log_density[self.old_bin] - self.log_density[new_bin])
            .exp()
        
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use rand_pcg::Pcg64Mcg;
    use rand::SeedableRng;
    use crate::examples::coin_flips::*;
    #[test]
    #[cfg_attr(miri,ignore)]
    fn wl_simulations_equal() {
        let mut rng = Pcg64Mcg::seed_from_u64(2239790);
        let ensemble = CoinFlipSequence::new(100, Pcg64Mcg::from_rng(&mut rng).unwrap());
        let histogram = HistogramFast::new_inclusive(0, 100).unwrap();
        let mut wl= WangLandau1T::new(
            0.0075,
            ensemble,
            rng,
            1,
            histogram,
            30
        ).unwrap();

        wl.init_mixed_heuristik(
            NonZeroUsize::new(3).unwrap(),
            6400i16,
            |e|  {
                Some(e.head_count())
            },
            None
        ).unwrap();

        let mut wl_backup = wl.clone();
        let start_wl= std::time::Instant::now();
        wl.wang_landau_convergence(
            |e| Some(e.head_count())
        );
        let dur_1 = start_wl.elapsed();
        let start_wl_acc = std::time::Instant::now();
        wl_backup.wang_landau_convergence_acc(
            CoinFlipSequence::update_head_count
        );
        let dur_2 = start_wl_acc.elapsed();
        println!("WL: {}, WL_ACC: {}, difference: {}", dur_1.as_nanos(), dur_2.as_nanos(), dur_1.as_nanos() - dur_2.as_nanos());

        // assert, that the different methods lead to the same result
        for (&log_value, &log_value_acc) in wl.log_density().iter().zip(wl_backup.log_density().iter()){
            assert_eq!(log_value, log_value_acc);
        }
    }
}
