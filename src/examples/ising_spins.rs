use rand::{Rng, distributions::{Bernoulli, Distribution}};
use std::ops::Mul;

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};


#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum Spin{
    Up,
    Down
}

impl Mul for Spin{
    type Output = f64;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.eq(&rhs) {
            1.0
        } else {
            -1.0
        }
    }
}

#[derive(Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct IsingLattice<R> {
    width: usize,
    height: usize,
    n: usize,
    j: f64,
    rng: R,
    up_prob: f64,
    spins: Vec<Spin>,
}


impl<R> IsingLattice<R> 
    where R: Rng
{   

    /// j is the coupling
    pub fn new(width: usize, height: usize, up_probability: f64, j: f64, mut rng: R) -> Option<Self>
    {
        if up_probability > 1.0 || up_probability < 0.0 {
            return None;
        }
        let n = width * height;
        if n == 0 {
            return None;
        }
        let bernoulli = Bernoulli::new(up_probability).unwrap();
        
        let mut spins = Vec::with_capacity(n);
        spins.extend(
            (0..n).map(|_| {
                if bernoulli.sample(&mut rng)
                {
                    Spin::Up
                } else {
                    Spin::Down
                }
            })
        );

        Some(
            Self{
                width,
                height,
                n,
                rng,
                up_prob: up_probability,
                spins,
                j
            }
        )        
    }

    pub fn energy(&self) -> f64
    {
        let mut e = 0.0;
        for y in 0..self.height {
            for x in 0..self.width {
                let s1 = self.spins[self.index(x, y)];
                IsingLaticeNeigborIter::new(self, x, y)
                    .for_each(
                        |s2| e += s1 * s2
                    );
            }
        }
        -e * 0.5
    }

    #[inline(always)]
    fn index(&self, x: usize, y: usize) -> usize
    {
        y * self.width + x
    }
}

#[derive(Clone)]
pub struct IsingLaticeNeigborIter<'a>{
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    spins: &'a [Spin],
    counter: u8,
}

impl<'a> IsingLaticeNeigborIter<'a>
{
    pub fn new<R>(ising_latice: &'a IsingLattice<R>, x: usize, y: usize) -> Self
    {
        Self{
            x,
            y,
            width: ising_latice.width,
            height: ising_latice.height,
            counter: 0,
            spins: &ising_latice.spins,
        }
    } 
}

impl<'a> Iterator for IsingLaticeNeigborIter<'a>
{
    type Item = Spin;
    // oben - links - rechts - unten
    fn next(&mut self) -> Option<Self::Item> {
        let (x, y) = match self.counter {
            0 =>  {
                let y = match self.y.checked_sub(1)
                {
                    Some(val) => val,
                    None => self.height - 1,
                };
                (self.x, y)
            }, 
            1 => {
                let x = match self.x.checked_sub(1) {
                    Some(val) => val,
                    None => self.width - 1,
                };
                (x, self.y)
            },
            2 => {
                let x = (self.x + 1) % self.width;
                (x, self.y)
            },
            3 => {
                let y = (self.y + 1) % self.height;
                (self.x, y)
            },
            _ => return None,
        };
        self.counter += 1;

        Some(
            self.spins[y * self.width + x]
        )
    }
}