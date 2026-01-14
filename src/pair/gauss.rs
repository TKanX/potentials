//! # Gaussian Potential
//!
//! A soft Gaussian potential used in coarse-grained models and GEM
//! (Gaussian Electrostatic Model) force fields.
//!
//! ## Formula
//!
//! ```text
//! V(r) = A * exp(-B * r^2)
//! ```
//!
//! where:
//! - `A`: Amplitude (energy units)
//! - `B`: Gaussian width parameter (1/length^2 units)
//!
//! ## Force Factor
//!
//! ```text
//! S = -(dV/dr) / r = 2 * A * B * exp(-B * r^2)
//! ```
//!
//! ## Implementation Notes
//!
//! - Stores `-B` internally for efficient `exp(-B*r^2)` computation
//! - Bounded: V(0) = A, V(∞) = 0
//! - Smooth: Infinitely differentiable
//! - Purely repulsive (if A > 0) or attractive (if A < 0)

use crate::base::Potential2;
use crate::math::Vector;

/// Gaussian potential.
///
/// ## Parameters
///
/// - `a`: Amplitude (energy units)
/// - `b`: Width parameter (1/length^2 units)
///
/// ## Precomputed Values
///
/// - `neg_b`: Stores `-b` internally for efficient `exp(-b*r^2)` computation
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gauss<T> {
    /// Amplitude
    a: T,
    /// Width parameter (stored negative for efficiency)
    neg_b: T,
}

impl<T: Vector> Gauss<T> {
    /// Creates a new Gaussian potential.
    ///
    /// ## Arguments
    ///
    /// - `a`: Amplitude (energy units)
    /// - `b`: Width parameter (1/length^2 units)
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::pair::Gauss;
    ///
    /// // Repulsive Gaussian
    /// let gauss = Gauss::<f64>::new(1.0, 0.5);
    /// ```
    #[inline]
    pub fn new(a: f64, b: f64) -> Self {
        Self {
            a: T::splat(a),
            neg_b: T::splat(-b),
        }
    }

    /// Creates a Gaussian potential from sigma parameterization.
    ///
    /// ```text
    /// V(r) = A * exp(-r^2 / (2 * sigma^2))
    /// ```
    ///
    /// ## Arguments
    ///
    /// - `a`: Amplitude
    /// - `sigma`: Gaussian width
    #[inline]
    pub fn from_sigma(a: f64, sigma: f64) -> Self {
        let b = 1.0 / (2.0 * sigma * sigma);
        Self::new(a, b)
    }

    /// Returns the amplitude.
    #[inline]
    pub fn amplitude(&self) -> T {
        self.a
    }
}

impl<T: Vector> Potential2<T> for Gauss<T> {
    /// Computes the potential energy.
    ///
    /// ```text
    /// V(r) = A * exp(-B * r^2)
    /// ```
    #[inline(always)]
    fn energy(&self, r_sq: T) -> T {
        // exp(-B * r^2) = exp(neg_b * r_sq)
        let exp_term = (self.neg_b * r_sq).exp();
        self.a * exp_term
    }

    /// Computes the force factor.
    ///
    /// ```text
    /// dV/dr = A * exp(-B*r^2) * (-2*B*r)
    /// S = -(dV/dr)/r = 2*A*B * exp(-B*r^2)
    /// ```
    #[inline(always)]
    fn force_factor(&self, r_sq: T) -> T {
        let exp_term = (self.neg_b * r_sq).exp();

        // S = -2 * neg_b * A * exp_term (since neg_b = -B)
        let two = T::splat(2.0);
        T::zero() - two * self.neg_b * self.a * exp_term
    }

    /// Computes energy and force factor together (optimized).
    ///
    /// Shares the computation of `exp_term` and `a_exp`.
    #[inline(always)]
    fn energy_force(&self, r_sq: T) -> (T, T) {
        let exp_term = (self.neg_b * r_sq).exp();
        let a_exp = self.a * exp_term;

        let energy = a_exp;

        let two = T::splat(2.0);
        let force = T::zero() - two * self.neg_b * a_exp;

        (energy, force)
    }
}
