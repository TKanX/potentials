//! # Quartic Bond Potential
//!
//! A fourth-order polynomial bond potential used in Class II force fields.
//!
//! ## Formula
//!
//! ```text
//! V(r) = k2 * (r - r0)^2 + k3 * (r - r0)^3 + k4 * (r - r0)^4
//! ```
//!
//! where:
//! - `k2`: Quadratic force constant (energy/length^2 units)
//! - `k3`: Cubic correction (energy/length^3 units)
//! - `k4`: Quartic correction (energy/length^4 units)
//! - `r0`: Equilibrium bond length (length units)
//!
//! ## Force Factor
//!
//! ```text
//! S = -(dV/dr) / r = -(2*k2*(r-r0) + 3*k3*(r-r0)^2 + 4*k4*(r-r0)^3) / r
//! ```
//!
//! ## Implementation Notes
//!
//! - Used in Class II force fields (CFF, PCFF, COMPASS)
//! - Better reproduces vibrational frequencies and anharmonic effects

use crate::base::Potential2;
use crate::math::Vector;

/// Quartic bond potential.
///
/// ## Parameters
///
/// - `k2`: Quadratic force constant (energy/length^2 units)
/// - `k3`: Cubic correction (energy/length^3 units)
/// - `k4`: Quartic correction (energy/length^4 units)
/// - `r0`: Equilibrium distance (length units)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quart<T> {
    /// Quadratic coefficient
    k2: T,
    /// Cubic coefficient
    k3: T,
    /// Quartic coefficient
    k4: T,
    /// Equilibrium distance
    r0: T,
}

impl<T: Vector> Quart<T> {
    /// Creates a new quartic bond potential.
    ///
    /// ## Arguments
    ///
    /// - `k2`: Quadratic force constant
    /// - `k3`: Cubic correction
    /// - `k4`: Quartic correction
    /// - `r0`: Equilibrium bond length
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::bond::Quart;
    ///
    /// // Class II force field parameters
    /// let bond = Quart::<f64>::new(300.0, -50.0, 10.0, 1.54);
    /// ```
    #[inline]
    pub fn new(k2: f64, k3: f64, k4: f64, r0: f64) -> Self {
        Self {
            k2: T::splat(k2),
            k3: T::splat(k3),
            k4: T::splat(k4),
            r0: T::splat(r0),
        }
    }

    /// Creates a quartic potential with only quadratic and quartic terms.
    ///
    /// This is symmetric about the equilibrium.
    #[inline]
    pub fn symmetric(k2: f64, k4: f64, r0: f64) -> Self {
        Self::new(k2, 0.0, k4, r0)
    }

    /// Returns the quadratic coefficient.
    #[inline]
    pub fn k2(&self) -> T {
        self.k2
    }

    /// Returns the cubic coefficient.
    #[inline]
    pub fn k3(&self) -> T {
        self.k3
    }

    /// Returns the quartic coefficient.
    #[inline]
    pub fn k4(&self) -> T {
        self.k4
    }

    /// Returns the equilibrium distance.
    #[inline]
    pub fn r0(&self) -> T {
        self.r0
    }
}

impl<T: Vector> Potential2<T> for Quart<T> {
    /// Computes the potential energy.
    ///
    /// ```text
    /// V(r) = k2*(r-r0)^2 + k3*(r-r0)^3 + k4*(r-r0)^4
    /// ```
    #[inline(always)]
    fn energy(&self, r_sq: T) -> T {
        let r = r_sq.sqrt();
        let dr = r - self.r0;
        let dr_sq = dr * dr;

        // Use Horner's method: dr^2 * (k2 + dr * (k3 + dr * k4))
        let inner = self.k3 + dr * self.k4;
        let result = self.k2 + dr * inner;

        dr_sq * result
    }

    /// Computes the force factor.
    ///
    /// ```text
    /// dV/dr = 2*k2*(r-r0) + 3*k3*(r-r0)^2 + 4*k4*(r-r0)^3
    /// S = -(dV/dr)/r
    /// ```
    #[inline(always)]
    fn force_factor(&self, r_sq: T) -> T {
        let r = r_sq.sqrt();
        let r_inv = r.recip();
        let dr = r - self.r0;

        let two = T::splat(2.0);
        let three = T::splat(3.0);
        let four = T::splat(4.0);

        // dV/dr = dr * (2*k2 + dr * (3*k3 + dr * 4*k4))
        let inner = three * self.k3 + dr * four * self.k4;
        let dv_dr = dr * (two * self.k2 + dr * inner);

        T::zero() - dv_dr * r_inv
    }

    /// Computes energy and force factor together (optimized).
    ///
    /// Shares the computation of `r`, `r_inv`, `dr`, and `dr_sq`.
    #[inline(always)]
    fn energy_force(&self, r_sq: T) -> (T, T) {
        let r = r_sq.sqrt();
        let r_inv = r.recip();
        let dr = r - self.r0;
        let dr_sq = dr * dr;

        let two = T::splat(2.0);
        let three = T::splat(3.0);
        let four = T::splat(4.0);

        // Energy: dr^2 * (k2 + dr * (k3 + dr * k4))
        let e_inner = self.k3 + dr * self.k4;
        let energy = dr_sq * (self.k2 + dr * e_inner);

        // Force: -dr * (2*k2 + dr * (3*k3 + dr * 4*k4)) / r
        let f_inner = three * self.k3 + dr * four * self.k4;
        let dv_dr = dr * (two * self.k2 + dr * f_inner);
        let force = T::zero() - dv_dr * r_inv;

        (energy, force)
    }
}
