//! # Dihedral Torsion Potentials
//!
//! Four-body potentials for dihedral angle rotation.
//!
//! ## Included Potentials
//!
//! | Potential | Description | Common Use |
//! |-----------|-------------|------------|
//! | [`Cos`] | Periodic cosine | AMBER, DREIDING |
//! | [`Opls`] | Fourier series | OPLS |
//! | [`Rb`] | Ryckaert-Bellemans | GROMOS |
//! | [`Harm`] | Harmonic | Restraints |
//!
//! ## Dihedral Convention
//!
//! For atoms i-j-k-l:
//! - phi is the angle between planes (i,j,k) and (j,k,l)
//! - phi = 0 when all four atoms are in the same plane (cis configuration)
//! - phi = 180° for trans configuration
//!
//! The sign convention follows the IUPAC definition:
//! - Looking along j->k, phi is positive for clockwise rotation of l relative to i
//!
//! ## Example
//!
//! ```
//! use potentials::torsion::{Cos, Potential4};
//!
//! // Ethane-like torsion: barrier = 2.5 kcal/mol, 3-fold symmetry
//! let torsion = Cos::new(2.5, 3, 0.0);
//!
//! let phi = std::f64::consts::PI / 3.0; // 60 degrees
//! let (cos_phi, sin_phi) = (phi.cos(), phi.sin());
//! let energy = torsion.energy(cos_phi, sin_phi);
//! ```

mod cos;
mod harm;
mod opls;
mod rb;

pub use cos::Cos;
pub use harm::Harm;
pub use opls::Opls;
pub use rb::Rb;

// Re-export base trait for convenience
pub use crate::base::Potential4;
