//! # Bonded Stretch Potentials
//!
//! Two-body potentials for covalent bond stretching.
//!
//! ## Included Potentials
//!
//! | Potential | Description | Common Use |
//! |-----------|-------------|------------|
//! | [`Harm`] | Harmonic k(r-r0)^2 | General, AMBER |
//! | [`G96`] | GROMOS96 k(r^2-r0^2)^2 | Fast approximation |
//! | [`Morse`] | D(1-exp(-a(r-r0)))^2 | Bond breaking |
//! | [`Fene`] | FENE -k*R^2*ln(1-(r/R)^2) | Polymers |
//! | [`Cubic`] | Cubic anharmonic | Small anharmonicity |
//! | [`Quart`] | Quartic | Class II force fields |
//!
//! ## Example
//!
//! ```
//! use potentials::bond::{Harm, Potential2};
//!
//! // C-C bond: k = 300 kcal/mol/A^2, r0 = 1.54 A
//! let bond = Harm::new(300.0, 1.54);
//!
//! let r_sq = 1.6 * 1.6;
//! let (energy, force_factor) = bond.energy_force(r_sq);
//! ```

mod cubic;
mod fene;
mod g96;
mod harm;
mod morse;
mod quart;

pub use cubic::Cubic;
pub use fene::Fene;
pub use g96::G96;
pub use harm::Harm;
pub use morse::Morse;
pub use quart::Quart;

// Re-export base trait for convenience
pub use crate::base::Potential2;
