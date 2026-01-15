//! # Harmonic Bond Potential
//!
//! The standard harmonic spring potential for bond stretching.
//!
//! ## Formula
//!
//! ```text
//! V(r) = k * (r - r0)^2
//! ```
//!
//! where:
//! - `k`: Force constant (energy/length^2 units)
//! - `r0`: Equilibrium bond length (length units)
//!
//! ## Force Factor
//!
//! ```text
//! S = -(dV/dr) / r = 2 * k * (r0/r - 1)
//! ```
//!
//! ## Implementation Notes
//!
//! - Some force fields use `V = (1/2) * k * (r - r0)^2`
//! - To convert: `k_here = k_other / 2`
//! - This is the most common bond potential (AMBER, CHARMM, OPLS)

use crate::base::Potential2;
use crate::math::Vector;

/// Harmonic bond potential.
///
/// ## Parameters
///
/// - `k`: Force constant (energy/length^2 units)
/// - `r0`: Equilibrium distance (length units)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Harm<T> {
    /// Force constant
    k: T,
    /// Equilibrium distance
    r0: T,
}

impl<T: Vector> Harm<T> {
    /// Creates a new harmonic bond potential.
    ///
    /// ## Arguments
    ///
    /// - `k`: Force constant (energy/length^2 units)
    /// - `r0`: Equilibrium bond length (length units)
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::bond::Harm;
    ///
    /// // Typical C-C single bond
    /// let bond = Harm::<f64>::new(300.0, 1.54);
    /// ```
    #[inline]
    pub fn new(k: f64, r0: f64) -> Self {
        Self {
            k: T::splat(k),
            r0: T::splat(r0),
        }
    }

    /// Returns the force constant.
    #[inline]
    pub fn k(&self) -> T {
        self.k
    }

    /// Returns the equilibrium distance.
    #[inline]
    pub fn r0(&self) -> T {
        self.r0
    }
}

impl<T: Vector> Potential2<T> for Harm<T> {
    /// Computes the potential energy.
    ///
    /// ```text
    /// V(r) = k * (r - r0)^2
    /// ```
    #[inline(always)]
    fn energy(&self, r_sq: T) -> T {
        let r = r_sq.sqrt();
        let dr = r - self.r0;
        self.k * dr * dr
    }

    /// Computes the force factor.
    ///
    /// ```text
    /// dV/dr = 2 * k * (r - r0)
    /// S = -(dV/dr)/r = -2 * k * (r - r0) / r
    ///   = 2 * k * (r0/r - 1)
    /// ```
    #[inline(always)]
    fn force_factor(&self, r_sq: T) -> T {
        let r = r_sq.sqrt();
        let two = T::splat(2.0);
        two * self.k * (self.r0 / r - T::one())
    }

    /// Computes energy and force factor together (optimized).
    ///
    /// Shares the computation of `r` and `r_inv`.
    #[inline(always)]
    fn energy_force(&self, r_sq: T) -> (T, T) {
        let r = r_sq.sqrt();
        let r_inv = r.recip();
        let dr = r - self.r0;

        let energy = self.k * dr * dr;

        let two = T::splat(2.0);
        let force = two * self.k * (self.r0 * r_inv - T::one());

        (energy, force)
    }
}
