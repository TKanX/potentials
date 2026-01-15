//! # Harmonic Improper Torsion
//!
//! The simplest improper potential for maintaining planarity.
//!
//! ## Formula
//!
//! ```text
//! V(xi) = k * (xi - xi0)^2
//! ```
//!
//! where:
//! - `xi`: Current improper angle (radians)
//! - `xi0`: Equilibrium angle (usually 0 for planar, ±35.26° for tetrahedral)
//! - `k`: Force constant (energy/radian² units)
//!
//! ## Derivative
//!
//! ```text
//! dV/d(xi) = 2 * k * (xi - xi0)
//! ```
//!
//! ## Implementation Notes
//!
//! - Used for aromatic rings and peptide bonds (xi0 = 0)
//! - Used for sp3 chirality (xi0 ≈ ±35.26°)
//! - Compatible with CHARMM, GROMACS, and other force fields

use crate::base::Potential4;
use crate::math::Vector;

/// Harmonic improper torsion potential.
///
/// ## Parameters
///
/// - `k`: Force constant (energy/radian² units)
/// - `xi0`: Equilibrium improper angle (radians)
///
/// ## Precomputed Values
///
/// - `two_k`: Stores `2*k` for efficient derivative computation
/// - `cos_xi0`, `sin_xi0`: Precomputed for angle-difference formula
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Harm<T> {
    two_k: T,
    xi0: T,
    cos_xi0: T,
    sin_xi0: T,
}

impl<T: Vector> Harm<T> {
    /// Creates a new harmonic improper potential.
    ///
    /// ## Arguments
    ///
    /// - `k`: Force constant (energy/radian² units)
    /// - `xi0`: Equilibrium angle (radians)
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::imp::Harm;
    ///
    /// // Planar constraint for aromatic ring
    /// let planar = Harm::<f64>::planar(40.0);
    ///
    /// // Chiral center constraint
    /// let chiral = Harm::<f64>::tetrahedral(40.0);
    /// ```
    #[inline]
    pub fn new(k: f64, xi0: f64) -> Self {
        Self {
            two_k: T::splat(2.0 * k),
            xi0: T::splat(xi0),
            cos_xi0: T::splat(xi0.cos()),
            sin_xi0: T::splat(xi0.sin()),
        }
    }

    /// Creates for planar geometry (xi0 = 0).
    #[inline]
    pub fn planar(k: f64) -> Self {
        Self::new(k, 0.0)
    }

    /// Creates for tetrahedral geometry (xi0 = 35.26°).
    ///
    /// The tetrahedral angle from the base is arcsin(1/sqrt(3)) ≈ 35.26°
    #[inline]
    pub fn tetrahedral(k: f64) -> Self {
        use crate::math::scalar_f64;
        // arcsin(1/sqrt(3)) = 0.6154797... rad ≈ 35.26°
        let xi0 = scalar_f64::asin(scalar_f64::sqrt(1.0 / 3.0));
        Self::new(k, xi0)
    }
}

impl<T: Vector> Potential4<T> for Harm<T> {
    /// Computes the potential energy.
    ///
    /// ```text
    /// V = k * (xi - xi0)^2
    /// ```
    #[inline(always)]
    fn energy(&self, cos_xi: T, sin_xi: T) -> T {
        let sin_delta = sin_xi * self.cos_xi0 - cos_xi * self.sin_xi0;
        let cos_delta = cos_xi * self.cos_xi0 + sin_xi * self.sin_xi0;

        let delta_xi = sin_delta.atan2(cos_delta);

        let half = T::splat(0.5);
        half * self.two_k * delta_xi * delta_xi
    }

    /// Computes dV/d(xi).
    ///
    /// ```text
    /// dV/d(xi) = 2k * (xi - xi0)
    /// ```
    #[inline(always)]
    fn derivative(&self, cos_xi: T, sin_xi: T) -> T {
        let sin_delta = sin_xi * self.cos_xi0 - cos_xi * self.sin_xi0;
        let cos_delta = cos_xi * self.cos_xi0 + sin_xi * self.sin_xi0;

        let delta_xi = sin_delta.atan2(cos_delta);

        self.two_k * delta_xi
    }

    /// Computes energy and derivative together (optimized).
    ///
    /// Shares the computation of `sin_delta`, `cos_delta`, and `delta_xi`.
    #[inline(always)]
    fn energy_derivative(&self, cos_xi: T, sin_xi: T) -> (T, T) {
        let sin_delta = sin_xi * self.cos_xi0 - cos_xi * self.sin_xi0;
        let cos_delta = cos_xi * self.cos_xi0 + sin_xi * self.sin_xi0;

        let delta_xi = sin_delta.atan2(cos_delta);

        let half = T::splat(0.5);
        let energy = half * self.two_k * delta_xi * delta_xi;
        let derivative = self.two_k * delta_xi;

        (energy, derivative)
    }
}
