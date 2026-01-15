//! # Non-Bonded Pair Potentials
//!
//! Two-body interaction potentials for non-bonded atoms.
//!
//! ## Included Potentials
//!
//! | Potential | Description | Common Use |
//! |-----------|-------------|------------|
//! | [`Lj`] | Lennard-Jones 12-6 | Van der Waals |
//! | [`Mie`] | Generalized Mie n-m | Tunable VdW |
//! | [`Buck`] | Buckingham (Exp-6) | Ionic systems |
//! | [`Coul`] | Coulomb 1/r | Electrostatics |
//! | [`Yukawa`] | Screened Coulomb | Ionic screening |
//! | [`Gauss`] | Gaussian | Soft matter (GEM) |
//! | [`Soft`] | Soft sphere | Purely repulsive |
//!
//! ## Example
//!
//! ```
//! use potentials::pair::{Lj, Potential2};
//!
//! // Argon LJ parameters: sigma = 3.4 A, epsilon = 0.238 kcal/mol
//! let lj = Lj::new(0.238, 3.4);
//!
//! let r_sq = 4.0 * 4.0; // Distance of 4 A
//! let energy = lj.energy(r_sq);
//! let force_factor = lj.force_factor(r_sq);
//! ```

mod buck;
mod coul;
mod gauss;
mod lj;
mod mie;
mod soft;
mod yukawa;

pub use buck::Buck;
pub use coul::Coul;
pub use gauss::Gauss;
pub use lj::Lj;
pub use mie::Mie;
pub use soft::Soft;
pub use yukawa::Yukawa;

// Re-export base trait for convenience
pub use crate::base::Potential2;
