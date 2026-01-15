//! # Harmonic Angle Potential (Theta Form)
//!
//! The standard harmonic potential in angle space.
//!
//! ## Formula
//!
//! ```text
//! V(theta) = k * (theta - theta0)^2
//! ```
//!
//! where:
//! - `k`: Force constant (energy/radian^2 units)
//! - `theta`: Actual bond angle (radians)
//! - `theta0`: Equilibrium angle (radians)
//!
//! ## Derivative Convention
//!
//! Returns dV/d(cos_theta), not dV/d(theta), for direct use in force calculation.
//!
//! ## Implementation Notes
//!
//! - Used by AMBER, OPLS, CHARMM
//! - Requires `acos` to compute theta from cos(theta)
//! - For better performance, consider [`Cos`](super::Cos)

use crate::base::Potential3;
use crate::math::Vector;

/// Harmonic angle potential in theta space.
///
/// ## Parameters
///
/// - `k`: Force constant (energy/radian^2 units)
/// - `theta0`: Equilibrium angle (radians)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Harm<T> {
    /// Force constant
    k: T,
    /// Equilibrium angle (radians)
    theta0: T,
}

impl<T: Vector> Harm<T> {
    /// Creates a new harmonic angle potential.
    ///
    /// ## Arguments
    ///
    /// - `k`: Force constant (energy/radian^2 units)
    /// - `theta0`: Equilibrium angle in radians
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::angle::Harm;
    ///
    /// // Water angle: 104.5 degrees
    /// let theta0 = 104.5 * std::f64::consts::PI / 180.0;
    /// let angle = Harm::<f64>::new(100.0, theta0);
    /// ```
    #[inline]
    pub fn new(k: f64, theta0: f64) -> Self {
        Self {
            k: T::splat(k),
            theta0: T::splat(theta0),
        }
    }

    /// Creates from theta0 in degrees (convenience).
    #[inline]
    pub fn from_degrees(k: f64, theta0_deg: f64) -> Self {
        let theta0_rad = theta0_deg * core::f64::consts::PI / 180.0;
        Self::new(k, theta0_rad)
    }

    /// Returns the force constant.
    #[inline]
    pub fn k(&self) -> T {
        self.k
    }

    /// Returns the equilibrium angle.
    #[inline]
    pub fn theta0(&self) -> T {
        self.theta0
    }
}

impl<T: Vector> Potential3<T> for Harm<T> {
    /// Computes the potential energy.
    ///
    /// ```text
    /// V(theta) = k * (theta - theta0)^2
    /// ```
    ///
    /// ## Note
    ///
    /// Requires `acos` to convert cos_theta to theta.
    #[inline(always)]
    fn energy(&self, _r_ij_sq: T, _r_jk_sq: T, cos_theta: T) -> T {
        let theta = cos_theta.acos();
        let dtheta = theta - self.theta0;
        self.k * dtheta * dtheta
    }

    /// Computes dV/d(cos_theta).
    ///
    /// ```text
    /// dV/d(theta) = 2 * k * (theta - theta0)
    /// d(theta)/d(cos_theta) = -1 / sin(theta) = -1 / sqrt(1 - cos^2)
    /// dV/d(cos_theta) = -2 * k * (theta - theta0) / sin(theta)
    /// ```
    #[inline(always)]
    fn derivative(&self, _r_ij_sq: T, _r_jk_sq: T, cos_theta: T) -> T {
        let theta = cos_theta.acos();
        let dtheta = theta - self.theta0;

        // sin(theta) = sqrt(1 - cos^2)
        // For numerical stability near 0 and 180 degrees
        let one = T::one();
        let sin_sq = one - cos_theta * cos_theta;
        let sin_theta = sin_sq.max(T::splat(1e-10)).sqrt();

        let two = T::splat(2.0);
        T::zero() - two * self.k * dtheta / sin_theta
    }

    /// Computes energy and derivative together (optimized).
    ///
    /// Shares the computation of `theta`, `dtheta`, and `sin_theta`.
    #[inline(always)]
    fn energy_derivative(&self, r_ij_sq: T, r_jk_sq: T, cos_theta: T) -> (T, T) {
        let theta = cos_theta.acos();
        let dtheta = theta - self.theta0;

        let energy = self.k * dtheta * dtheta;

        let one = T::one();
        let sin_sq = one - cos_theta * cos_theta;
        let sin_theta = sin_sq.max(T::splat(1e-10)).sqrt();

        let two = T::splat(2.0);
        let derivative = T::zero() - two * self.k * dtheta / sin_theta;

        // Suppress unused warnings
        let _ = (r_ij_sq, r_jk_sq);

        (energy, derivative)
    }
}
