//! # Cosine Angle Potential
//!
//! A harmonic potential in cosine space, avoiding expensive `acos` operations.
//!
//! ## Formula
//!
//! ```text
//! V(theta) = k * (cos(theta) - cos(theta0))^2
//! ```
//!
//! where:
//! - `k`: Force constant (energy units)
//! - `cos(theta)`: Actual angle cosine
//! - `cos(theta0)`: Equilibrium angle cosine
//!
//! ## Derivative
//!
//! ```text
//! dV/d(cos_theta) = 2 * k * (cos_theta - cos0)
//! ```
//!
//! ## Implementation Notes
//!
//! - **Much faster** than theta-based harmonic: no `acos` needed
//! - Cosine is directly available from dot product
//! - Used by GROMOS and DREIDING force fields
//! - Near equilibrium: k_cos = k_theta * sin^2(theta0)

use crate::base::Potential3;
use crate::math::Vector;

/// Harmonic cosine angle potential.
///
/// ## Parameters
///
/// - `k`: Force constant (energy units)
/// - `cos0`: Equilibrium angle cosine
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cos<T> {
    /// Force constant
    k: T,
    /// Equilibrium cosine
    cos0: T,
}

impl<T: Vector> Cos<T> {
    /// Creates a new cosine angle potential.
    ///
    /// ## Arguments
    ///
    /// - `k`: Force constant (energy units)
    /// - `cos0`: Cosine of equilibrium angle
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::angle::Cos;
    ///
    /// // Tetrahedral angle: 109.5 degrees
    /// let cos0 = (109.5_f64 * std::f64::consts::PI / 180.0).cos();
    /// let angle = Cos::<f64>::new(100.0, cos0);
    /// ```
    #[inline]
    pub fn new(k: f64, cos0: f64) -> Self {
        Self {
            k: T::splat(k),
            cos0: T::splat(cos0),
        }
    }

    /// Creates from equilibrium angle in radians.
    #[inline]
    pub fn from_theta(k: f64, theta0: f64) -> Self {
        Self::new(k, theta0.cos())
    }

    /// Creates from equilibrium angle in degrees.
    #[inline]
    pub fn from_degrees(k: f64, theta0_deg: f64) -> Self {
        let theta0_rad = theta0_deg * core::f64::consts::PI / 180.0;
        Self::from_theta(k, theta0_rad)
    }

    /// Returns the force constant.
    #[inline]
    pub fn k(&self) -> T {
        self.k
    }

    /// Returns the equilibrium cosine.
    #[inline]
    pub fn cos0(&self) -> T {
        self.cos0
    }
}

impl<T: Vector> Potential3<T> for Cos<T> {
    /// Computes the potential energy.
    ///
    /// ```text
    /// V = k * (cos(theta) - cos0)^2
    /// ```
    ///
    /// No `acos` required!
    #[inline(always)]
    fn energy(&self, _r_ij_sq: T, _r_jk_sq: T, cos_theta: T) -> T {
        let delta = cos_theta - self.cos0;
        self.k * delta * delta
    }

    /// Computes dV/d(cos_theta).
    ///
    /// ```text
    /// dV/d(cos_theta) = 2 * k * (cos_theta - cos0)
    /// ```
    #[inline(always)]
    fn derivative(&self, _r_ij_sq: T, _r_jk_sq: T, cos_theta: T) -> T {
        let delta = cos_theta - self.cos0;
        let two = T::splat(2.0);
        two * self.k * delta
    }

    /// Computes energy and derivative together (optimized).
    ///
    /// Shares the computation of `delta`.
    #[inline(always)]
    fn energy_derivative(&self, _r_ij_sq: T, _r_jk_sq: T, cos_theta: T) -> (T, T) {
        let delta = cos_theta - self.cos0;
        let two = T::splat(2.0);

        let energy = self.k * delta * delta;
        let derivative = two * self.k * delta;

        (energy, derivative)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::PI;

    #[test]
    fn test_cos_at_equilibrium() {
        let cos0 = 0.5;
        let angle: Cos<f64> = Cos::new(100.0, cos0);

        let energy = angle.energy(1.0, 1.0, cos0);
        assert_relative_eq!(energy, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cos_derivative_at_equilibrium() {
        let cos0 = 0.5;
        let angle: Cos<f64> = Cos::new(100.0, cos0);

        let deriv = angle.derivative(1.0, 1.0, cos0);
        assert_relative_eq!(deriv, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cos_displaced() {
        let k = 100.0;
        let cos0 = 0.5;
        let angle: Cos<f64> = Cos::new(k, cos0);

        let cos_theta = 0.7;
        let energy = angle.energy(1.0, 1.0, cos_theta);

        let delta = cos_theta - cos0;
        let expected = k * delta * delta;

        assert_relative_eq!(energy, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_cos_from_constructors() {
        let theta0 = PI / 3.0;
        let cos0 = theta0.cos();

        let a1: Cos<f64> = Cos::new(100.0, cos0);
        let a2: Cos<f64> = Cos::from_theta(100.0, theta0);
        let a3: Cos<f64> = Cos::from_degrees(100.0, 60.0);

        let cos_theta = 0.3;
        assert_relative_eq!(
            a1.energy(1.0, 1.0, cos_theta),
            a2.energy(1.0, 1.0, cos_theta),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            a1.energy(1.0, 1.0, cos_theta),
            a3.energy(1.0, 1.0, cos_theta),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_cos_numerical_derivative() {
        let angle: Cos<f64> = Cos::new(100.0, 0.5);
        let cos_theta = 0.3;

        let h = 1e-7;
        let e_plus = angle.energy(1.0, 1.0, cos_theta + h);
        let e_minus = angle.energy(1.0, 1.0, cos_theta - h);
        let deriv_numerical = (e_plus - e_minus) / (2.0 * h);

        let deriv_analytical = angle.derivative(1.0, 1.0, cos_theta);

        assert_relative_eq!(deriv_analytical, deriv_numerical, epsilon = 1e-6);
    }

    #[test]
    fn test_cos_energy_derivative_consistency() {
        let angle: Cos<f64> = Cos::new(100.0, 0.5);
        let cos_theta = 0.3;

        let (e1, d1) = angle.energy_derivative(1.0, 1.0, cos_theta);
        let e2 = angle.energy(1.0, 1.0, cos_theta);
        let d2 = angle.derivative(1.0, 1.0, cos_theta);

        assert_relative_eq!(e1, e2, epsilon = 1e-10);
        assert_relative_eq!(d1, d2, epsilon = 1e-10);
    }
}
