//! # Improper Torsion Potentials
//!
//! Improper dihedrals are used to maintain planarity of sp2 centers
//! (like carbonyl carbons, aromatic rings) and chirality at
//! stereocenters.
//!
//! ## Available Types
//!
//! | Type | Description | Formula |
//! |------|-------------|---------|
//! | [`Harm`] | Harmonic | `k * (xi - xi0)^2` |
//! | [`Cos`] | Periodic cosine | `k * (1 + cos(n*xi - d))` |
//! | [`Dist`] | Distance-based | `k * (r - r0)^2` (out-of-plane) |
//!
//! ## Improper vs Proper Dihedrals
//!
//! - Proper: A-B-C-D measures rotation around B-C bond
//! - Improper: Central atom + 3 neighbors, measures deviation from plane

mod cos;
mod dist;
mod harm;

pub use cos::Cos;
pub use dist::Dist;
pub use harm::Harm;

// Re-export base trait for convenience
pub use crate::base::Potential4;
