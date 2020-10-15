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
pub struct Metropolis<E, R, S, Res>
{
    ensemble: E,
    rng: R,
    energy: f64,
    m_beta: f64,
    step_size: usize,
    counter: usize,
    step_target: usize,
    marker_s: PhantomData<S>,
    marker_res: PhantomData<Res>,
}

impl<R, E, S, Res> Metropolis<E, R, S, Res>
{
    pub fn new_from_m_beta(
        rng: R,
        ensemble: E,
        energy: f64,
        m_beta: f64,
        step_size: usize,
        step_target: usize,
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
                step_target,
                step_size,
            }
        )
    }

    pub fn new_from_temperature(
        rng: R,
        ensemble: E,
        energy: f64,
        temperature: f64,
        step_size: usize,
        step_target: usize,
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
            step_target
        )
    }

    /// returns true if step target was reached
    pub fn is_finished(&self) -> bool
    {
        self.step_target <= self.counter
    }

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

    /// returns, how many steps should be performed in total
    pub fn step_target(&self) -> usize {
        self.step_target
    }
}

impl<E, R, S, Res> Metropolis<E, R, S, Res>
    where R: Rng,
    E: MarkovChain<S, Res>
{

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
    /// # Explanation
    /// * Performes markov chain using the markov chain trait
    ///
    /// * Let the current state of the system be S(i) with corresponding energy `E(i) = energy_fn(S(i))`.
    /// * Now perform a markov step, such that the new system is S_new with energy E_new.
    /// * The new state will be accepted (meaning S(i+1) = S_new) with probability:
    /// `min[1.0, exp{-1/T * (E_new - E(i))}]`
    /// * otherwise the new state will be rejected, meaning S(i + 1) = S(i).
    /// Afterwards, `measure` is called.
    pub fn metropolis<Energy, Mes>(
        &mut self,
        mut energy_fn: Energy,
        measure: Mes,
    )
        where Energy: FnMut(&E) -> Option<f64>,
        Mes: Fn(&Self), 
    {
        for _ in self.counter..=self.step_target
        {
            self.metropolis_step(&mut energy_fn);
            measure(self);
        }
    }

    pub unsafe fn metropolis_unsafe<Energy, Mes>(
        &mut self,
        mut energy_fn: Energy,
        mut measure: Mes,
    )
        where Energy: FnMut(&mut E) -> Option<f64>,
        Mes: FnMut(&Self), 
    {
        for _ in self.counter..=self.step_target
        {
            self.metropolis_step_unsafe(&mut energy_fn);
            measure(self);
        }
    }

    pub fn metropolis_efficient<Energy, Mes>
    (
        &mut self,
        mut energy_fn: Energy,
        mut measure: Mes,
    )
        where Energy: FnMut(&E, f64, &[S]) -> Option<f64>,
        Mes: FnMut(&Self), 
    {
        for _ in self.counter..=self.step_target
        {
            self.metropolis_step_efficient(&mut energy_fn);
            measure(self);
        }
    }

    pub unsafe fn metropolis_efficient_unsafe<Energy, Mes>
    (
        &mut self,
        mut energy_fn: Energy,
        mut measure: Mes,
    )
        where Energy: Fn(&mut E, f64, &[S]) -> Option<f64>,
        Mes: FnMut(&mut Self), 
    {
        for _ in self.counter..=self.step_target
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