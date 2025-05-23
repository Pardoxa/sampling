
/// # Create a markov chain by doing markov steps
pub trait MarkovChain<S, Res> {
    /// * undo a markov step, return result-state
    /// * if you want to undo more than one step
    ///   see [`undo_steps`](#method.undo_steps)
    fn undo_step(&mut self, step: &S) -> Res;

    /// * undo a markov, **panic** on invalid result state
    /// * for undoing multiple steps see [`undo_steps_quiet`](#method.undo_steps_quiet)
    fn undo_step_quiet(&mut self, step: &S);

    /// # Markov step
    /// * use this to perform a markov step step
    /// * for doing multiple markov steps at once, use [`m_steps`](#method.m_steps)
    fn m_step(&mut self) -> S;

    /// #  Markov steps
    /// * use this to perform multiple markov steps at once
    /// * steps can be used to undo the steps with `self.undo_steps(steps)`
    /// * `steps` will be emptied before step is performed
    #[inline]
    fn m_steps(&mut self, count: usize, steps: &mut Vec<S>) {
        steps.clear();
        steps.extend((0..count)
            .map(|_| self.m_step())
        );
    }

    /// # Markov steps without return
    /// * use this to perform multiple markov steps at once
    /// * only use this if you **know** that you do **not** want to undo the steps
    /// * you cannot undo this steps, but therefore it does not need to allocate a vector 
    ///   for undoing steps
    fn m_steps_quiet(&mut self, count: usize)
    {
        for _ in 0..count {
            self.m_step();
        }
    }

    /// # Accumulating markov step
    /// * this calculates something while performing the markov chain, e.g., the current energy,
    ///   which can be more efficient then calculating it from scratch afterwards
    #[inline]
    fn m_step_acc<Acc, AccFn>(&mut self, acc: &mut Acc, mut acc_fn: AccFn) -> S
    where AccFn: FnMut(&Self, &S, &mut Acc)
    {
        let s = self.m_step();
        acc_fn(self, &s, acc);
        s
    }

    /// # Accumulating markov steps
    /// * this calculates something while performing the markov chain, e.g., the current energy
    ///   which can be more efficient then calculating it from scratch afterwards
    #[inline]
    fn m_steps_acc<Acc, AccFn>
    (
        &mut self,
        count: usize,
        steps: &mut Vec<S>,
        acc: &mut Acc,
        mut acc_fn: AccFn
    )
    where AccFn: FnMut(&Self, &S, &mut Acc)
    {
        steps.clear();
        steps.extend(
            (0..count)
                .map(|_| self.m_step_acc(acc, &mut acc_fn))
        );
    }

    /// # Accumulating markov steps
    /// * this calculates something while performing the markov chain, e.g., the current energy
    ///   which can be more efficient then calculating it from scratch afterwards
    /// * quiet step, i.e., you will not be able to undo the step
    #[inline]
    fn m_steps_acc_quiet<Acc, AccFn>(&mut self, count: usize, acc: &mut Acc, mut acc_fn: AccFn)
    where AccFn: FnMut(&Self, &S, &mut Acc)
    {
        for _ in 0..count{
            let _ = self.m_step_acc(acc, &mut acc_fn);
        }
    }

    /// # Undo markov steps
    /// * Note: uses undo_step in correct order and returns result
    /// ## Important:
    /// * look at specific implementation of `undo_step`, every thing mentioned there applies to each step
    fn undo_steps(&mut self, steps: &[S], res: &mut Vec<Res>) {
        res.clear();
        res.extend(
            steps.iter()
                .rev()
                .map(|step| self.undo_step(step))
        );
    }

    /// # Undo markov steps
    /// * Note: uses `undo_step_quiet` in correct order
    /// ## Important:
    /// * look at specific implementation of `undo_step_quiet`, every thing mentioned there applies to each step
    fn undo_steps_quiet(&mut self, steps: &[S]) {
        steps.iter()
            .rev()
            .for_each( |step| self.undo_step_quiet(step));
    }


    /// # Function called whenever the steps are accepted.
    /// *You can use it to create the acceptance statistics if 
    /// you move a variety of different moves
    /// * If you use the default implementation this will be a optimized out
    #[inline]
    fn steps_accepted(&mut self, _steps: &[S])
    {

    }

    /// # Function called whenever the steps are rejected.
    /// * You can use it to create the acceptance statistics if 
    ///   you move a variety of different moves
    /// * If you use the default implementation this will be a optimized out
    #[inline]
    fn steps_rejected(&mut self, _steps: &[S])
    {

    }
}

/// For easy sampling of your ensemble
pub trait SimpleSample{
    /// # Randomizes self according to  model
    /// * this is intended for creation of initial sample
    /// * used in [`simple_sample`](#method.simple_sample)
    ///   and [`simple_sample_vec`](#method.simple_sample_vec)
    fn randomize(&mut self);

    /// # do the following `times` times:
    /// 1) `f(self)`
    /// 2) `self.randomize()`
    fn simple_sample<F>(&mut self, times: usize, mut f: F)
        where F: FnMut(&Self)
    {
        for _ in 0..times {
            f(self);
            self.randomize();
        }
    }

    /// # do the following `times` times:
    /// 1) `res = f(self)`
    /// 2) `self.randomize()`
    /// ## res is collected into Vector
    fn simple_sample_vec<F, G>(&mut self, times: usize, mut f: F) -> Vec<G>
        where F: FnMut(&Self) -> G
    {
        let mut vec = Vec::with_capacity(times);
        for _ in 0..times {
            vec.push(f(self));
            self.randomize();
        }
        vec
    }
}

/// # Access internal random number generator
pub trait HasRng<Rng>
where Rng: rand::Rng
{
    /// # Access RNG
    /// If, for some reason, you want access to the internal random number generator: Here you go
    fn rng(&mut self) -> &mut Rng;

    /// # If you need to exchange the internal rng
    /// * swaps internal random number generator with `rng`
    fn swap_rng(&mut self, rng: &mut Rng);
}