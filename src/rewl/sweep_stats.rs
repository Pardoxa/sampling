use std::time::*;

#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SweepStats {
    buf: Vec<Duration>,
    counter: usize,
}

const BUFFER_SIZE: usize = 511;

impl SweepStats {
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(BUFFER_SIZE + 1),
            counter: 0,
        }
    }

    pub fn push(&mut self, duration: Duration) {
        if self.buf.len() <= BUFFER_SIZE {
            self.buf.push(duration);
        } else {
            self.buf[self.counter] = duration;
            if self.counter == BUFFER_SIZE {
                self.counter = 0;
            } else {
                self.counter += 1;
            }
        }
    }

    pub fn buf(&self) -> &[Duration] {
        &self.buf
    }

    pub fn averag_duration(&self) -> Duration {
        if self.buf.len() == 0 {
            return Duration::from_micros(0);
        }
        let sum: Duration = self.buf.iter().sum();
        sum / self.buf.len() as u32
    }

    pub fn percent_high_low(&self) -> (Duration, Duration) {
        if self.buf.len() < 2 {
            return (Duration::from_micros(0), Duration::from_micros(0));
        }
        let mut b = self.buf.clone();
        b.sort_unstable();
        let low = (b.len() as f64 * 0.9).floor() as usize;
        let high = (b.len() as f64 * 0.1).ceil() as usize;
        (b[low], b[high])
    }
}
