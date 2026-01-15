//! # Cubic Bond Potential
//!
//! Harmonic potential with a cubic anharmonic correction term.
//!
//! ## Formula
//!
//! ```text
//! V(r) = k * (r - r0)^2 + k_cubic * (r - r0)^3
//! ```
//!
//! where:
//! - `k`: Quadratic force constant (energy/length^2 units)
//! - `k_cubic`: Cubic correction (energy/length^3 units)
//! - `r0`: Equilibrium bond length (length units)
//!
//! ## Force Factor
//!
//! ```text
//! S = -(dV/dr) / r = -(2*k*(r-r0) + 3*k_cubic*(r-r0)^2) / r
//! ```
//!
//! ## Implementation Notes
//!
//! - Asymmetric: stretching and compression have different energies
//! - `k_cubic < 0`: softer for stretching, stiffer for compression (typical)
//! - Unbound for large deformations; use only for small displacements

use crate::base::Potential2;
use crate::math::Vector;

/// Cubic anharmonic bond potential.
///
/// ## Parameters
///
/// - `k`: Quadratic force constant (energy/length^2 units)
/// - `k_cubic`: Cubic correction (energy/length^3 units)
/// - `r0`: Equilibrium distance (length units)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cubic<T> {
    /// Quadratic force constant
    k: T,
    /// Cubic correction
    k_cubic: T,
    /// Equilibrium distance
    r0: T,
}

impl<T: Vector> Cubic<T> {
    /// Creates a new cubic bond potential.
    ///
    /// ## Arguments
    ///
    /// - `k`: Quadratic force constant (energy/length^2)
    /// - `k_cubic`: Cubic correction (energy/length^3)
    /// - `r0`: Equilibrium bond length (length units)
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::bond::Cubic;
    ///
    /// // Bond with slight anharmonic correction
    /// let bond = Cubic::<f64>::new(300.0, -50.0, 1.54);
    /// ```
    #[inline]
    pub fn new(k: f64, k_cubic: f64, r0: f64) -> Self {
        Self {
            k: T::splat(k),
            k_cubic: T::splat(k_cubic),
            r0: T::splat(r0),
        }
    }

    /// Returns the quadratic force constant.
    #[inline]
    pub fn k(&self) -> T {
        self.k
    }

    /// Returns the cubic correction.
    #[inline]
    pub fn k_cubic(&self) -> T {
        self.k_cubic
    }

    /// Returns the equilibrium distance.
    #[inline]
    pub fn r0(&self) -> T {
        self.r0
    }
}

impl<T: Vector> Potential2<T> for Cubic<T> {
    /// Computes the potential energy.
    ///
    /// ```text
    /// V(r) = k * (r - r0)^2 + k_cubic * (r - r0)^3
    /// ```
    #[inline(always)]
    fn energy(&self, r_sq: T) -> T {
        let r = r_sq.sqrt();
        let dr = r - self.r0;
        let dr_sq = dr * dr;
        let dr_cube = dr_sq * dr;

        self.k * dr_sq + self.k_cubic * dr_cube
    }

    /// Computes the force factor.
    ///
    /// ```text
    /// dV/dr = 2*k*(r-r0) + 3*k_cubic*(r-r0)^2
    /// S = -(dV/dr)/r = -(2*k*(r-r0) + 3*k_cubic*(r-r0)^2) / r
    /// ```
    #[inline(always)]
    fn force_factor(&self, r_sq: T) -> T {
        let r = r_sq.sqrt();
        let r_inv = r.recip();
        let dr = r - self.r0;

        let two = T::splat(2.0);
        let three = T::splat(3.0);

        let dv_dr = two * self.k * dr + three * self.k_cubic * dr * dr;
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

        let energy = self.k * dr_sq + self.k_cubic * dr_sq * dr;
        let dv_dr = two * self.k * dr + three * self.k_cubic * dr_sq;
        let force = T::zero() - dv_dr * r_inv;

        (energy, force)
    }
}
