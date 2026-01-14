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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gauss_at_zero() {
        let gauss: Gauss<f64> = Gauss::new(5.0, 1.0);
        let energy = gauss.energy(0.0);
        assert_relative_eq!(energy, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_decay() {
        let a = 1.0;
        let b = 0.5;
        let gauss: Gauss<f64> = Gauss::new(a, b);

        let r = 2.0;
        let r_sq = r * r;
        let energy = gauss.energy(r_sq);
        let expected = a * (-b * r_sq).exp();

        assert_relative_eq!(energy, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_force_at_zero() {
        let a = 3.0;
        let b = 0.5;
        let gauss: Gauss<f64> = Gauss::new(a, b);

        let force = gauss.force_factor(1e-20);
        let expected = 2.0 * a * b;

        assert_relative_eq!(force, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_from_sigma() {
        let a = 2.0;
        let sigma = 1.5;
        let g1: Gauss<f64> = Gauss::from_sigma(a, sigma);
        let g2: Gauss<f64> = Gauss::new(a, 1.0 / (2.0 * sigma * sigma));

        let r_sq = 3.0;
        assert_relative_eq!(g1.energy(r_sq), g2.energy(r_sq), epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_energy_force_consistency() {
        let gauss: Gauss<f64> = Gauss::new(10.0, 0.25);
        let r_sq = 2.25;

        let (e1, f1) = gauss.energy_force(r_sq);
        let e2 = gauss.energy(r_sq);
        let f2 = gauss.force_factor(r_sq);

        assert_relative_eq!(e1, e2, epsilon = 1e-10);
        assert_relative_eq!(f1, f2, epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_numerical_derivative() {
        let gauss: Gauss<f64> = Gauss::new(10.0, 0.25);
        let r = 1.5;
        let r_sq = r * r;

        let h = 1e-6;
        let v_plus = gauss.energy((r + h) * (r + h));
        let v_minus = gauss.energy((r - h) * (r - h));
        let dv_dr_numerical = (v_plus - v_minus) / (2.0 * h);

        let s_numerical = -dv_dr_numerical / r;
        let s_analytical = gauss.force_factor(r_sq);

        assert_relative_eq!(s_analytical, s_numerical, epsilon = 1e-6);
    }
}
