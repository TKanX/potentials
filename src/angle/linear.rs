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
