//! # Yukawa Potential (Screened Coulomb)
//!
//! The Yukawa potential models screened electrostatic interactions,
//! commonly used for charged particles in ionic solutions.
//!
//! ## Formula
//!
//! ```text
//! V(r) = A * exp(-kappa * r) / r
//! ```
//!
//! where:
//! - `A`: Amplitude (energy * length units)
//! - `kappa`: Inverse Debye screening length (1/length units)
//!
//! ## Force Factor
//!
//! ```text
//! S = -(dV/dr) / r = A * exp(-kappa * r) * (kappa * r + 1) / r^3
//! ```
//!
//! ## Implementation Notes
//!
//! - Stores `-kappa` internally for efficient `exp(-kappa*r)` computation
//! - `kappa -> 0`: Reduces to bare Coulomb (1/r)
//! - `kappa -> ∞`: Very short-ranged interaction

use crate::base::Potential2;
use crate::math::Vector;

/// Yukawa (screened Coulomb) potential.
///
/// ## Parameters
///
/// - `a`: Amplitude (energy * length units)
/// - `kappa`: Inverse screening length (1/length units)
///
/// ## Precomputed Values
///
/// - `neg_kappa`: Stores `-kappa` internally for efficient `exp(-kappa*r)` computation
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yukawa<T> {
    /// Amplitude
    a: T,
    /// Inverse screening length (stored negative for efficiency)
    neg_kappa: T,
}

impl<T: Vector> Yukawa<T> {
    /// Creates a new Yukawa potential.
    ///
    /// ## Arguments
    ///
    /// - `a`: Amplitude (energy * length units)
    /// - `kappa`: Inverse screening length (1/length units)
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::pair::Yukawa;
    ///
    /// // Screened Coulomb with Debye length of 10 A
    /// let yukawa = Yukawa::<f64>::new(332.0, 0.1);
    /// ```
    #[inline]
    pub fn new(a: f64, kappa: f64) -> Self {
        Self {
            a: T::splat(a),
            neg_kappa: T::splat(-kappa),
        }
    }

    /// Creates a Yukawa potential from Debye length.
    ///
    /// ## Arguments
    ///
    /// - `a`: Amplitude
    /// - `debye_length`: Screening length (kappa = 1/debye_length)
    #[inline]
    pub fn from_debye_length(a: f64, debye_length: f64) -> Self {
        Self::new(a, 1.0 / debye_length)
    }

    /// Returns the amplitude.
    #[inline]
    pub fn amplitude(&self) -> T {
        self.a
    }
}

impl<T: Vector> Potential2<T> for Yukawa<T> {
    /// Computes the potential energy.
    ///
    /// ```text
    /// V(r) = A * exp(-kappa * r) / r
    /// ```
    #[inline(always)]
    fn energy(&self, r_sq: T) -> T {
        let r = r_sq.sqrt();
        let r_inv = r.recip();
        let exp_term = (self.neg_kappa * r).exp();

        self.a * exp_term * r_inv
    }

    /// Computes the force factor.
    ///
    /// ```text
    /// dV/dr = A * exp(-kappa*r) * (-kappa/r - 1/r^2)
    ///       = A * exp(-kappa*r) * (-kappa*r - 1) / r^2
    /// S = -(dV/dr)/r = A * exp(-kappa*r) * (kappa*r + 1) / r^3
    /// ```
    #[inline(always)]
    fn force_factor(&self, r_sq: T) -> T {
        let r = r_sq.sqrt();
        let r_inv = r.recip();
        let exp_term = (self.neg_kappa * r).exp();

        // kappa * r + 1 (note: neg_kappa stores -kappa)
        let kappa_r_plus_1 = T::one() - self.neg_kappa * r;

        self.a * exp_term * kappa_r_plus_1 * r_inv * r_inv * r_inv
    }

    /// Computes energy and force factor together (optimized).
    ///
    /// Shares the computation of `r`, `r_inv`, `exp_term`, and `a_exp`.
    #[inline(always)]
    fn energy_force(&self, r_sq: T) -> (T, T) {
        let r = r_sq.sqrt();
        let r_inv = r.recip();
        let exp_term = (self.neg_kappa * r).exp();
        let a_exp = self.a * exp_term;

        let energy = a_exp * r_inv;

        let kappa_r_plus_1 = T::one() - self.neg_kappa * r;
        let force = a_exp * kappa_r_plus_1 * r_inv * r_inv * r_inv;

        (energy, force)
    }
}
