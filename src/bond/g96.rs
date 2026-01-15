//! # GROMOS96 Bond Potential
//!
//! The GROMOS96 quartic bond potential, which avoids expensive
//! square root operations by working directly with r^2.
//!
//! ## Formula
//!
//! ```text
//! V(r) = (k/4) * (r^2 - r0^2)^2
//! ```
//!
//! where:
//! - `k`: Force constant (energy/length^4 units)
//! - `r0`: Equilibrium bond length (length units)
//!
//! ## Force Factor
//!
//! ```text
//! S = -(dV/dr) / r = -k * (r^2 - r0^2)
//! ```
//!
//! ## Implementation Notes
//!
//! - Operates entirely on r^2, avoiding sqrt and division by r
//! - Near equilibrium: V_G96 ≈ k * r0^2 * (r - r0)^2
//! - Relation to harmonic: k_G96 = k_harm / r0^2

use crate::base::Potential2;
use crate::math::Vector;

/// GROMOS96 bond potential.
///
/// ## Parameters
///
/// - `k`: Force constant (energy/length^4 units)
/// - `r0`: Equilibrium distance (length units)
///
/// ## Precomputed Values
///
/// - `k_quarter`: Stores `k/4` for efficient computation
/// - `r0_sq`: Stores `r0^2` to avoid sqrt
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct G96<T> {
    /// Force constant divided by 4
    k_quarter: T,
    /// Equilibrium distance squared
    r0_sq: T,
}

impl<T: Vector> G96<T> {
    /// Creates a new GROMOS96 bond potential.
    ///
    /// ## Arguments
    ///
    /// - `k`: Force constant (energy/length^4 units)
    /// - `r0`: Equilibrium bond length (length units)
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::bond::G96;
    ///
    /// // Convert from harmonic: k_g96 = k_harm / r0^2
    /// let k_harm = 300.0;
    /// let r0 = 1.54;
    /// let bond = G96::<f64>::new(k_harm / (r0 * r0), r0);
    /// ```
    #[inline]
    pub fn new(k: f64, r0: f64) -> Self {
        Self {
            k_quarter: T::splat(k / 4.0),
            r0_sq: T::splat(r0 * r0),
        }
    }

    /// Creates from r0 squared directly (avoids a multiplication).
    #[inline]
    pub fn from_r0_sq(k: f64, r0_sq: f64) -> Self {
        Self {
            k_quarter: T::splat(k / 4.0),
            r0_sq: T::splat(r0_sq),
        }
    }

    /// Returns the equilibrium distance squared.
    #[inline]
    pub fn r0_sq(&self) -> T {
        self.r0_sq
    }
}

impl<T: Vector> Potential2<T> for G96<T> {
    /// Computes the potential energy.
    ///
    /// ```text
    /// V(r) = (k/4) * (r^2 - r0^2)^2
    /// ```
    ///
    /// Note: No sqrt required!
    #[inline(always)]
    fn energy(&self, r_sq: T) -> T {
        let delta = r_sq - self.r0_sq;
        self.k_quarter * delta * delta
    }

    /// Computes the force factor.
    ///
    /// ```text
    /// dV/dr = (k/4) * 2 * (r^2 - r0^2) * 2r = k * r * (r^2 - r0^2)
    /// S = -(dV/dr)/r = -k * (r^2 - r0^2)
    /// ```
    ///
    /// Note: No sqrt or division required!
    #[inline(always)]
    fn force_factor(&self, r_sq: T) -> T {
        let delta = r_sq - self.r0_sq;
        let four = T::splat(4.0);
        T::zero() - four * self.k_quarter * delta
    }

    /// Computes energy and force factor together (optimized).
    ///
    /// Shares the computation of `delta`.
    #[inline(always)]
    fn energy_force(&self, r_sq: T) -> (T, T) {
        let delta = r_sq - self.r0_sq;
        let four = T::splat(4.0);

        let energy = self.k_quarter * delta * delta;
        let force = T::zero() - four * self.k_quarter * delta;

        (energy, force)
    }
}
