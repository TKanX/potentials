//! # Coulomb Potential
//!
//! The basic Coulomb electrostatic interaction.
//!
//! ## Formula
//!
//! ```text
//! V(r) = k * q1 * q2 / r
//! ```
//!
//! where:
//! - `k`: Coulomb constant (depends on unit system)
//! - `q1, q2`: Particle charges
//! - `r`: Inter-particle distance
//!
//! ## Force Factor
//!
//! ```text
//! S = -(dV/dr) / r = k * q1 * q2 / r^3
//! ```
//!
//! ## Unit Systems
//!
//! | Units | k value | Energy | Length |
//! |-------|---------|--------|--------|
//! | Real  | 332.064 | kcal/mol | Angstrom |
//! | Metal | 14.3996 | eV | Angstrom |
//! | SI    | 8.988e9 | J | m |
//!
//! ## Implementation Notes
//!
//! This implementation stores the product `k * q1 * q2` as a single coefficient.
//! For efficiency in molecular dynamics, this should be precomputed per pair.
//!
//! For periodic systems, consider using Ewald summation or reaction field
//! methods instead of bare Coulomb.

use crate::base::Potential2;
use crate::math::Vector;

/// Coulomb (1/r) potential.
///
/// ## Parameters
///
/// - `kqq`: Product of Coulomb constant and charges: k * q1 * q2
///
/// Positive kqq = repulsion (like charges)
/// Negative kqq = attraction (opposite charges)
///
/// ## Precomputed Values
///
/// - `kqq`: Stores `k * q1 * q2` for efficient computation
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coul<T> {
    /// Combined coefficient: k * q1 * q2
    kqq: T,
}

impl<T: Vector> Coul<T> {
    /// Creates a new Coulomb potential from precomputed kqq.
    ///
    /// ## Arguments
    ///
    /// - `kqq`: Product k * q1 * q2 in consistent units
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::pair::Coul;
    /// use potentials::consts::COULOMB_REAL;
    ///
    /// // Na+ and Cl- interaction (real units)
    /// let q_na = 1.0;   // +1 e
    /// let q_cl = -1.0;  // -1 e
    /// let kqq = COULOMB_REAL * q_na * q_cl;
    /// let coul = Coul::<f64>::new(kqq);
    /// ```
    #[inline]
    pub fn new(kqq: f64) -> Self {
        Self { kqq: T::splat(kqq) }
    }

    /// Creates a Coulomb potential from charges and Coulomb constant.
    ///
    /// ## Arguments
    ///
    /// - `k`: Coulomb constant (e.g., [`COULOMB_REAL`](crate::consts::COULOMB_REAL))
    /// - `q1`: First particle charge
    /// - `q2`: Second particle charge
    #[inline]
    pub fn from_charges(k: f64, q1: f64, q2: f64) -> Self {
        Self::new(k * q1 * q2)
    }

    /// Returns the kqq coefficient.
    #[inline]
    pub fn kqq(&self) -> T {
        self.kqq
    }
}

impl<T: Vector> Potential2<T> for Coul<T> {
    /// Computes the potential energy.
    ///
    /// ```text
    /// V(r) = kqq / r = kqq / sqrt(r^2)
    /// ```
    #[inline(always)]
    fn energy(&self, r_sq: T) -> T {
        let r_inv = r_sq.rsqrt();
        self.kqq * r_inv
    }

    /// Computes the force factor.
    ///
    /// ```text
    /// dV/dr = -kqq / r^2
    /// S = -(dV/dr)/r = kqq / r^3
    /// ```
    #[inline(always)]
    fn force_factor(&self, r_sq: T) -> T {
        // r^-3 = r^-2 * r^-1 = r^-2 * rsqrt(r^2)
        let r_sq_inv = r_sq.recip();
        let r_inv = r_sq.rsqrt();
        self.kqq * r_sq_inv * r_inv
    }

    /// Computes energy and force factor together (optimized).
    ///
    /// Shares the computation of `r_sq_inv` and `r_inv`.
    #[inline(always)]
    fn energy_force(&self, r_sq: T) -> (T, T) {
        let r_sq_inv = r_sq.recip();
        let r_inv = r_sq.rsqrt();

        let energy = self.kqq * r_inv;
        let force = self.kqq * r_sq_inv * r_inv;

        (energy, force)
    }
}
