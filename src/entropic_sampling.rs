//! # For using entropic sampling after a wang landau simulation
//! You can store related measurable quantities for looking at correlations etc. 
mod entropic_adaptive;
mod entropic;
mod traits;

pub use entropic::*;
pub use entropic_adaptive::*;
pub use traits::*;