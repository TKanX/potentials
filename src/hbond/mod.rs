//! # Hydrogen Bond Potentials
//!
//! Specialized potentials for hydrogen bonding interactions.
//!
//! ## Available Types
//!
//! | Type | Description | Formula |
//! |------|-------------|---------|
//! | [`Dreid`] | DREIDING 12-10 | `D0 * (5*(R0/R)^12 - 6*(R0/R)^10) * cos^N(theta)` |
//! | [`LjCos`] | LJ with angular | `4*eps*((sigma/r)^12 - (sigma/r)^6) * cos^N(theta)` |
//!
//! Both use const generic `N` for compile-time optimization of the angular exponent.
//!
//! ## Notes
//!
//! These are "explicit" H-bond potentials. Many force fields treat
//! H-bonds implicitly through standard electrostatics and vdW.

mod dreid;
mod ljcos;

pub use dreid::Dreid;
pub use ljcos::LjCos;
