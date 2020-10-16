use rand::Rng;
use crate::{MarkovChain, HasRng};
use std::marker::PhantomData;

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

#[derive(Clone, Copy)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum MetropolisError
{
    /// Energy function for current state of ensemble returns None
    InvalidState,
    /// Invalid nan encountered
    NAN,
    /// m_beta cannot be infinitiy or minus infinity!
    InfinitBeta,
}

#[derive(Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
/// # Create a metropolis simulation
/// Citation see, e.g,
/// > M. E. J. Newman and G. T. Barkema, "Monte Carlo Methods in Statistical Physics"
///   *Clarendon Press*, 1999, ISBN:&nbsp;978-0-19-8517979
///
/// # Explanation
/// * used for large-deviation simulations
/// * Performes markov chain using the markov chain trait
/// ## All `self.metropolis*` functions do the following
/// * Let the current state of the system be S(i) with corresponding energy `E(i) = energy_fn(S(i))`.
/// * Now perform a markov step, such that the new system is S_new with energy E_new.
/// * The new state will be accepted (meaning S(i+1) = S_new) with probability:
/// `min[1.0, exp{-1/T * (E_new - E(i))}]`
/// * otherwise the new state will be rejected, meaning S(i + 1) = S(i).
/// * Afterwards the `measure` function is called
pub struct Metropolis<E, R, S, Res>
{
    ensemble: E,
    rng: R,
    energy: f64,
    m_beta: f64,
    step_size: usize,
    counter: usize,
    marker_s: PhantomData<S>,
    marker_res: PhantomData<Res>,
}

impl<R, E, S, Res> Metropolis<E, R, S, Res>
{

    /// returns stored `m_beta` value (-&beta; for metropolis)
    pub fn m_beta(&self) -> f64 {
        self.m_beta
    }

    /// sets m_beta (minus beta). Is related to the temperature: m_beta = -1 / temperature
    pub fn set_m_beta(&mut self, m_beta: f64)
    {
        self.m_beta = m_beta;
    }

    /// sets m_beta according to m_beta = -1 / temperature
    pub fn set_temperature(&mut self, temperature: f64)
    {
        self.m_beta = -1.0 / temperature;
    }

    /// returns stored value for `current_energy`
    pub fn energy(&self) -> f64 {
        self.energy
    }

    /// # set stored value for `current_energy`
    /// * Will return Err() if you try to set the energy to nan
    /// * otherwise it will set the stored `energy` and return Ok(())
    /// # Important
    /// * It is very unlikely that you need this function - Only use it, if you know what you are doing
    pub unsafe fn set_energy(&mut self, energy: f64) -> Result<(),()>{
        if energy.is_nan() {
            Err(())
        } else {
            self.energy = energy;
            Ok(())
        }
    }

    /// returns reference to ensemble
    pub fn ensemble(&self) -> &E
    {
        &self.ensemble
    }

    /// returns mutable reference to ensemble
    /// * use with care!
    /// * if you change your ensemble, this might invalidate
    /// the simulation!
    /// * The metropolis functions do not calculate the energy of the current state,
    pub unsafe fn ensemble_mut(&mut self) -> &mut E
    {
        &mut self.ensemble
    }

    /// returns stored value for the `counter`, i.e., where to resume iteration
    pub fn counter(&self) -> usize {
        self.counter
    }

    /// resets the `counter` to 0
    pub fn reset_counter(&mut self) {
        self.counter = 0;
    }

    /// return current `stepsize`
    pub fn step_size(&self) -> usize {
        self.step_size
    }

    /// * change the `stepsize`
    /// * returns err if you try to set stepsize to `0`, because that would be invalid
    pub fn set_step_size(&mut self, step_size: usize) -> Result<(),()>
    {
        if step_size == 0 {
            Err(())
        } else {
            self.step_size = step_size;
            Ok(())
        }
    }

}

impl<E, R, S, Res> Metropolis<E, R, S, Res>
    where R: Rng,
    E: MarkovChain<S, Res>
{

    /// # Create a new Metropolis struct - used for Metropolis simulations
    ///  
    /// |               | meaning                                                                      |
    /// |---------------|------------------------------------------------------------------------------|
    /// | `rng`         | the Rng used to decide, if a state should be accepted or rejected            |
    /// | `ensemble`    | the ensemble that is explored with the markov chain                           |
    /// | `energy`      | current energy of the ensemble - cannot be NAN, should match `energy_fn(ensemble)` (see `metropolis*` functions)  |
    /// | `m_beta`      | minus beta, has to be finite - used for acceptance, i.e., probability to accept a markov step from Energy E to Energy E_new is min[1.0, exp{m_beta * (E_new - E)}] |
    /// | `step_size`   | is used for each markov step, i.e., `ensemble.m_steps(stepsize)` is called   |
    /// 
    /// * will return Err if `energy` is nan or `m_beta` is not finite
    pub fn new_from_m_beta(
        rng: R,
        ensemble: E,
        energy: f64,
        m_beta: f64,
        step_size: usize,
    ) -> Result<Self, MetropolisError>
    {
        if energy.is_nan() || m_beta.is_nan() {
            return Err(MetropolisError::NAN);
        }
        if !m_beta.is_finite(){
            return Err(MetropolisError::InfinitBeta);
        }
        Ok(
            Self{
                ensemble,
                rng,
                energy,
                m_beta,
                marker_s: PhantomData::<S>,
                marker_res: PhantomData::<Res>,
                counter: 0,
                step_size,
            }
        )
    }

    /// # Create a new Metropolis struct - used for Metropolis simulations
    ///  
    /// |               | meaning                                                                      |
    /// |---------------|------------------------------------------------------------------------------|
    /// | `rng`         | the Rng used to decide, if a state should be accepted or rejected            |
    /// | `ensemble`    | the ensemble that is explored with the markov chain                           |
    /// | `energy`      | current energy of the ensemble - cannot be NAN, should match `energy_fn(ensemble)` (see `metropolis*` functions)  |
    /// | `temperature` | m_beta = -1.0/temperature. Used for acceptance, i.e., probability to accept a markov step from Energy E to Energy E_new is min[1.0, exp{m_beta * (E_new - E)}] |
    /// | `step_size`   | is used for each markov step, i.e., `ensemble.m_steps(stepsize)` is called   |
    /// 
    /// * will return Err if `energy` is nan or `m_beta` is not finite
    pub fn new_from_temperature(
        rng: R,
        ensemble: E,
        energy: f64,
        temperature: f64,
        step_size: usize,
    ) -> Result<Self, MetropolisError>
    {
        if temperature.is_nan() {
            return Err(MetropolisError::NAN);
        }
        Self::new_from_m_beta(
            rng,
            ensemble,
            energy,
            -1.0 / temperature,
            step_size,
        )
    }

    /// # Change, which markov chain is used for the metropolis simulations
    /// * Use this if there are different ways to perform a markov chain for your problem
    /// and you want to switch between them
    pub fn change_markov_chain<S2, Res2>(self) -> Metropolis<E, R, S2, Res2>
        where E: MarkovChain<S2, Res2>
    {
        Metropolis::<E, R, S2, Res2>{
            ensemble: self.ensemble,
            rng: self.rng,
            energy: self.energy,
            step_size: self.step_size,
            m_beta: self.m_beta,
            counter: self.counter,
            marker_s: PhantomData::<S2>,
            marker_res: PhantomData::<Res2>,
        }
    }

    /// Perform a single Metropolis step
    #[inline(always)]
    unsafe fn metropolis_step_unsafe<Energy>(&mut self, mut energy_fn: Energy)
    where Energy: FnMut(&mut E) -> Option<f64>
    {
        self.metropolis_step_efficient_unsafe(
            |ensemble, _, _|  energy_fn(ensemble) 
        )
    }

    #[inline(always)]
    fn metropolis_step<Energy>(&mut self, mut energy_fn: Energy)
    where Energy: FnMut(&E) -> Option<f64>
    {
        unsafe {
            self.metropolis_step_unsafe(
                |ensemble| energy_fn(ensemble)
            )
        }
    }

    /// Perform a single Metropolis step
    #[inline(always)]
    unsafe fn metropolis_step_efficient_unsafe<Energy>(&mut self, mut energy_fn: Energy)
    where Energy: FnMut(&mut E, f64, &[S]) -> Option<f64>
    {
        self.counter += 1;
        let step = self.ensemble.m_steps(self.step_size);
        let new_energy = match energy_fn(&mut self.ensemble, self.energy, &step) {
            None => {
                self.ensemble.undo_steps_quiet(step);
                return;
            },
            Some(e) => {
                e
            }
        };

        let a_prob = (self.m_beta * (new_energy - self.energy)).exp().min(1.0);

        let accepted = self.rng.gen_bool(a_prob);

        if accepted {
            self.energy = new_energy;
        } else {
            self.ensemble.undo_steps_quiet(step);
        }
    }

    /// Perform a single Metropolis step
    #[inline(always)]
    fn metropolis_step_efficient<Energy>(&mut self, mut energy_fn: Energy)
    where Energy: FnMut(&E, f64, &[S]) -> Option<f64>
    {
        unsafe {
            self.metropolis_step_efficient_unsafe(
                |ensemble, energy, steps| energy_fn(ensemble, energy, steps)
            )
        }
    }


    /// # Metropolis Simulation
    /// * [see](#all-selfmetropolis-functions-do-the-following)
    /// * performs `self.counter..=step_target` markov steps
    /// * `energy_fn(self.ensemble)` is assumed to equal `self.energy` at the beginning!
    /// * if `energy_fn` returns None, the step will always be rejected
    /// * after each acceptance/rejection, `measure` is called
    /// # Note
    /// * I assume, that the energy_fn never returns `nan`
    /// If nan is possible, please check for that beforhand and return `None` in that case
    /// * Maybe do the same for infinity, it is unlikely, that infinit energys make sense 
    pub fn metropolis<Energy, Mes>(
        &mut self,
        step_target: usize,
        mut energy_fn: Energy,
        mut measure: Mes,
    )
        where Energy: FnMut(&E) -> Option<f64>,
        Mes: FnMut(&Self), 
    {
        for _ in self.counter..=step_target
        {
            self.metropolis_step(&mut energy_fn);
            measure(self);
        }
    }

    /// # Metropolis Simulation
    /// * [see](#all-selfmetropolis-functions-do-the-following)
    /// * performs `self.counter..=step_target` markov steps
    /// * `energy_fn(self.ensemble)` is assumed to equal `self.energy` at the beginning!
    /// * if `energy_fn` returns None, the step will always be rejected
    /// * after each acceptance/rejection, `measure` is called
    /// # Important
    /// * if possible, prefere [`self.metropolis`](#method.metropolis) as it is safer
    /// * use this, if your energy function needs mutable access, or `measure`needs mutable access.
    ///  Be careful though, this can invalidate the results of your simulation
    /// # Note
    /// * I assume, that the energy_fn never returns `nan`
    /// If nan is possible, please check for that beforhand and return `None` in that case
    /// * Maybe do the same for infinity, it is unlikely, that infinit energys make sense 
    pub unsafe fn metropolis_unsafe<Energy, Mes>(
        &mut self,
        step_target: usize,
        mut energy_fn: Energy,
        mut measure: Mes,
    )
        where Energy: FnMut(&mut E) -> Option<f64>,
        Mes: FnMut(&mut Self), 
    {
        for _ in self.counter..=step_target
        {
            self.metropolis_step_unsafe(&mut energy_fn);
            measure(self);
        }
    }

    /// # Metropolis Simulation
    /// * [see](#all-selfmetropolis-functions-do-the-following)
    /// * performs `self.counter..=step_target` markov steps
    /// * `energy_fn(self.ensemble)` is assumed to equal `self.energy` at the beginning!
    /// * if `energy_fn` returns None, the step will always be rejected
    /// * after each acceptance/rejection, `measure` is called
    /// # Difference to [`self.metropolis`](#method.metropolis)
    /// * Function parameter of energy_fn: &ensemble, old_energy, &[steps] - that
    ///  means, you should prefere this, if you can calculate the new energy more efficient 
    /// by accessing the old energy and the information about what the markov step changed
    /// # Note
    /// * I assume, that the energy_fn never returns `nan`
    /// If nan is possible, please check for that beforhand and return `None` in that case
    /// * Maybe do the same for infinity, it is unlikely, that infinit energys make sense 
    pub fn metropolis_efficient<Energy, Mes>
    (
        &mut self,
        step_target: usize,
        mut energy_fn: Energy,
        mut measure: Mes,
    )
        where Energy: FnMut(&E, f64, &[S]) -> Option<f64>,
        Mes: FnMut(&Self), 
    {
        for _ in self.counter..=step_target
        {
            self.metropolis_step_efficient(&mut energy_fn);
            measure(self);
        }
    }

    pub unsafe fn metropolis_efficient_unsafe<Energy, Mes>
    (
        &mut self,
        step_target: usize,
        mut energy_fn: Energy,
        mut measure: Mes,
    )
        where Energy: Fn(&mut E, f64, &[S]) -> Option<f64>,
        Mes: FnMut(&mut Self), 
    {
        for _ in self.counter..=step_target
        {
            self.metropolis_step_efficient_unsafe(&mut energy_fn);
            measure(self);
        }
    }

    pub fn metropolis_while<Energy, Mes, Cond>(
        &mut self,
        mut energy_fn: Energy,
        mut measure: Mes,
        mut condition: Cond,
    )
        where Energy: FnMut(&E) -> Option<f64>,
        Mes: FnMut(&Self),
        Cond: FnMut(&Self) -> bool,
    {
        while condition(self) {
            self.metropolis_step(&mut energy_fn);
            measure(self);
        }
    }

    pub unsafe fn metropolis_while_unsafe<Energy, Mes, Cond>(
        &mut self,
        mut energy_fn: Energy,
        mut measure: Mes,
        mut condition: Cond,
    )
        where Energy: FnMut(&mut E) -> Option<f64>,
        Mes: FnMut(&mut Self),
        Cond: FnMut(&mut Self) -> bool,
    {
        while condition(self) {
            self.metropolis_step_unsafe(&mut energy_fn);
            measure(self);
        }
    }

    pub fn metropolis_efficient_while<Energy, Mes, Cond>(
        &mut self,
        mut energy_fn: Energy,
        mut measure: Mes,
        mut condition: Cond,
    )
        where Energy: FnMut(&E, f64, &[S]) -> Option<f64>,
        Mes: FnMut(&Self),
        Cond: FnMut(&Self) -> bool,
    {
        while condition(self) {
            self.metropolis_step_efficient(&mut energy_fn);
            measure(self);
        }
    }

    pub unsafe fn metropolis_efficient_while_unsafe<Energy, Mes, Cond>(
        &mut self,
        mut energy_fn: Energy,
        mut measure: Mes,
        mut condition: Cond,
    )
        where Energy: FnMut(&mut E, f64, &[S]) -> Option<f64>,
        Mes: FnMut(&mut Self),
        Cond: FnMut(&mut Self) -> bool,
    {
        while condition(self) {
            self.metropolis_step_efficient_unsafe(&mut energy_fn);
            measure(self);
        }
    }


}


impl<E, R, S, Res> HasRng<R> for Metropolis<E, R, S, Res>
    where R: Rng
{
    fn rng(&mut self) -> &mut R {
        &mut self.rng
    }

    fn swap_rng(&mut self, rng: &mut R) {
        std::mem::swap(&mut self.rng,rng);
    }
}