//! # Linear Angle Potential
//!
//! A simple cosine potential for maintaining linear geometry.
//!
//! ## Formula
//!
//! ```text
//! V(theta) = k * (1 + cos(theta))
//! ```
//!
//! where:
//! - `k`: Force constant (energy units)
//! - `theta`: Bond angle
//!
//! ## Derivative
//!
//! ```text
//! dV/d(cos_theta) = k
//! ```
//!
//! ## Implementation Notes
//!
//! - V(180°) = 0 (minimum for linear geometry)
//! - V(0°) = 2k (maximum for folded geometry)
//! - Used for sp hybridization (CO2, acetylene)

use crate::base::Potential3;
use crate::math::Vector;

/// Linear angle potential.
///
/// ## Parameters
///
/// - `k`: Force constant (energy units)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Linear<T> {
    /// Force constant
    k: T,
}

impl<T: Vector> Linear<T> {
    /// Creates a new linear angle potential.
    ///
    /// ## Arguments
    ///
    /// - `k`: Force constant (energy units)
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::angle::Linear;
    ///
    /// // Enforce linearity with strong force constant
    /// let angle = Linear::<f64>::new(200.0);
    /// ```
    #[inline]
    pub fn new(k: f64) -> Self {
        Self { k: T::splat(k) }
    }

    /// Returns the force constant.
    #[inline]
    pub fn k(&self) -> T {
        self.k
    }
}

impl<T: Vector> Potential3<T> for Linear<T> {
    /// Computes the potential energy.
    ///
    /// ```text
    /// V = k * (1 + cos(theta))
    /// ```
    #[inline(always)]
    fn energy(&self, _r_ij_sq: T, _r_jk_sq: T, cos_theta: T) -> T {
        self.k * (T::one() + cos_theta)
    }

    /// Computes dV/d(cos_theta).
    ///
    /// ```text
    /// dV/d(cos_theta) = k
    /// ```
    #[inline(always)]
    fn derivative(&self, _r_ij_sq: T, _r_jk_sq: T, _cos_theta: T) -> T {
        self.k
    }

    /// Computes energy and derivative together.
    #[inline(always)]
    fn energy_derivative(&self, _r_ij_sq: T, _r_jk_sq: T, cos_theta: T) -> (T, T) {
        let energy = self.k * (T::one() + cos_theta);
        let derivative = self.k;
        (energy, derivative)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_linear_at_180() {
        let k = 100.0;
        let linear: Linear<f64> = Linear::new(k);

        let energy = linear.energy(1.0, 1.0, -1.0);
        assert_relative_eq!(energy, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_linear_at_0() {
        let k = 100.0;
        let linear: Linear<f64> = Linear::new(k);

        let energy = linear.energy(1.0, 1.0, 1.0);
        assert_relative_eq!(energy, 2.0 * k, epsilon = 1e-10);
    }

    #[test]
    fn test_linear_at_90() {
        let k = 100.0;
        let linear: Linear<f64> = Linear::new(k);

        let energy = linear.energy(1.0, 1.0, 0.0);
        assert_relative_eq!(energy, k, epsilon = 1e-10);
    }

    #[test]
    fn test_linear_derivative_constant() {
        let k = 100.0;
        let linear: Linear<f64> = Linear::new(k);

        assert_relative_eq!(linear.derivative(1.0, 1.0, -1.0), k, epsilon = 1e-10);
        assert_relative_eq!(linear.derivative(1.0, 1.0, 0.0), k, epsilon = 1e-10);
        assert_relative_eq!(linear.derivative(1.0, 1.0, 1.0), k, epsilon = 1e-10);
    }

    #[test]
    fn test_linear_numerical_derivative() {
        let linear: Linear<f64> = Linear::new(100.0);
        let cos_theta = 0.3;

        let h = 1e-7;
        let e_plus = linear.energy(1.0, 1.0, cos_theta + h);
        let e_minus = linear.energy(1.0, 1.0, cos_theta - h);
        let deriv_numerical = (e_plus - e_minus) / (2.0 * h);

        let deriv_analytical = linear.derivative(1.0, 1.0, cos_theta);

        assert_relative_eq!(deriv_analytical, deriv_numerical, epsilon = 1e-6);
    }
}
