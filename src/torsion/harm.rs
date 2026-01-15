//! # Harmonic Torsion Potential
//!
//! A simple harmonic restraint in the dihedral angle itself.
//!
//! ## Formula
//!
//! ```text
//! V(phi) = k * (phi - phi0)^2
//! ```
//!
//! where:
//! - `phi`: Current dihedral angle (radians)
//! - `phi0`: Equilibrium angle (radians)
//! - `k`: Force constant (energy/radian² units)
//!
//! ## Derivative
//!
//! ```text
//! dV/d(phi) = 2 * k * (phi - phi0)
//! ```
//!
//! ## Implementation Notes
//!
//! - Used primarily for restraints or unusual geometries
//! - Reconstructs phi from cos/sin using angle-difference formula
//! - Handles full -π to +π range correctly

use crate::base::Potential4;
use crate::math::Vector;

/// Harmonic torsion potential (restraint).
///
/// ## Parameters
///
/// - `k`: Force constant (energy/radian² units)
/// - `phi0`: Equilibrium dihedral angle (radians)
///
/// ## Precomputed Values
///
/// - `two_k`: Stores `2*k` for efficient derivative computation
/// - `cos_phi0`, `sin_phi0`: Precomputed for angle-difference formula
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Harm<T> {
    two_k: T,
    phi0: T,
    cos_phi0: T,
    sin_phi0: T,
}

impl<T: Vector> Harm<T> {
    /// Creates a new harmonic torsion potential.
    ///
    /// ## Arguments
    ///
    /// - `k`: Force constant [energy/radian²]
    /// - `phi0`: Equilibrium angle (radians)
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::torsion::Harm;
    /// use core::f64::consts::PI;
    ///
    /// // Restrain to trans configuration (180°)
    /// let restraint = Harm::<f64>::new(100.0, PI);
    /// ```
    #[inline]
    pub fn new(k: f64, phi0: f64) -> Self {
        Self {
            two_k: T::splat(2.0 * k),
            phi0: T::splat(phi0),
            cos_phi0: T::splat(phi0.cos()),
            sin_phi0: T::splat(phi0.sin()),
        }
    }

    /// Creates with equilibrium at cis (phi0 = 0).
    #[inline]
    pub fn cis(k: f64) -> Self {
        Self::new(k, 0.0)
    }

    /// Creates with equilibrium at trans (phi0 = pi).
    #[inline]
    pub fn trans(k: f64) -> Self {
        Self::new(k, core::f64::consts::PI)
    }
}

impl<T: Vector> Potential4<T> for Harm<T> {
    /// Computes the potential energy.
    ///
    /// Uses the angle difference formula to compute (phi - phi0) from
    /// cos and sin values without explicitly computing atan2.
    ///
    /// ```text
    /// delta_phi = atan2(sin(phi - phi0), cos(phi - phi0))
    ///           = atan2(sin_phi*cos_phi0 - cos_phi*sin_phi0,
    ///                   cos_phi*cos_phi0 + sin_phi*sin_phi0)
    /// V = k * delta_phi^2
    /// ```
    #[inline(always)]
    fn energy(&self, cos_phi: T, sin_phi: T) -> T {
        // Angle difference using trig identities
        // sin(phi - phi0) = sin_phi * cos_phi0 - cos_phi * sin_phi0
        // cos(phi - phi0) = cos_phi * cos_phi0 + sin_phi * sin_phi0
        let sin_delta = sin_phi * self.cos_phi0 - cos_phi * self.sin_phi0;
        let cos_delta = cos_phi * self.cos_phi0 + sin_phi * self.sin_phi0;

        // Reconstruct angle using atan2
        let delta_phi = sin_delta.atan2(cos_delta);

        // V = k * delta^2 = (2k/2) * delta^2
        let half = T::splat(0.5);
        half * self.two_k * delta_phi * delta_phi
    }

    /// Computes dV/d(phi).
    ///
    /// ```text
    /// dV/d(phi) = 2k * (phi - phi0)
    /// ```
    #[inline(always)]
    fn derivative(&self, cos_phi: T, sin_phi: T) -> T {
        let sin_delta = sin_phi * self.cos_phi0 - cos_phi * self.sin_phi0;
        let cos_delta = cos_phi * self.cos_phi0 + sin_phi * self.sin_phi0;

        let delta_phi = sin_delta.atan2(cos_delta);

        // dV/dphi = 2k * delta
        self.two_k * delta_phi
    }

    /// Computes energy and derivative together (optimized).
    ///
    /// Shares the computation of `sin_delta`, `cos_delta`, and `delta_phi`.
    #[inline(always)]
    fn energy_derivative(&self, cos_phi: T, sin_phi: T) -> (T, T) {
        let sin_delta = sin_phi * self.cos_phi0 - cos_phi * self.sin_phi0;
        let cos_delta = cos_phi * self.cos_phi0 + sin_phi * self.sin_phi0;

        let delta_phi = sin_delta.atan2(cos_delta);

        let half = T::splat(0.5);
        let energy = half * self.two_k * delta_phi * delta_phi;
        let derivative = self.two_k * delta_phi;

        (energy, derivative)
    }
}
