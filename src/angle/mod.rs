//! # Angle Bending Potentials
//!
//! Three-body potentials for bond angle deformation.
//!
//! ## Included Potentials
//!
//! | Potential | Description | Common Use |
//! |-----------|-------------|------------|
//! | [`Harm`] | Harmonic k(theta-theta0)^2 | AMBER, OPLS |
//! | [`Cos`] | Cosine k(cos-cos0)^2 | GROMOS, DREIDING |
//! | [`Linear`] | Linear k(1+cos) | Linear molecules |
//! | [`Urey`] | Urey-Bradley (angle + 1-3 bond) | CHARMM |
//! | [`Cross`] | Bond-bond cross term | Class II |
//!
//! ## Coordinate Convention
//!
//! For angle i-j-k:
//! - j is the central (vertex) atom
//! - theta is the angle between vectors j->i and j->k
//! - cos(theta) = (r_ji · r_jk) / (|r_ji| |r_jk|)
//!
//! ## Example
//!
//! ```
//! use potentials::angle::{Cos, Potential3};
//!
//! // Water H-O-H angle: k = 100 kcal/mol, theta0 = 104.5 deg
//! let theta0_rad = 104.5 * std::f64::consts::PI / 180.0;
//! let angle = Cos::new(100.0, theta0_rad.cos());
//!
//! let cos_theta = 0.0; // 90 degrees
//! let deriv = angle.derivative(1.0, 1.0, cos_theta);
//! ```

mod cos;
mod cross;
mod harm;
mod linear;
mod urey;

pub use cos::Cos;
pub use cross::Cross;
pub use harm::Harm;
pub use linear::Linear;
pub use urey::Urey;

// Re-export base trait for convenience
pub use crate::base::Potential3;
