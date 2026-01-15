//! # Distance-Based Improper Potential
//!
//! Uses the out-of-plane distance to enforce planarity.
//!
//! ## Formula
//!
//! ```text
//! V(d) = k * (d - d0)^2
//! ```
//!
//! where:
//! - `d`: Signed out-of-plane distance of central atom from plane
//! - `d0`: Equilibrium distance (usually 0 for planar)
//! - `k`: Force constant (energy/length² units)
//!
//! ## Derivative
//!
//! ```text
//! dV/dd = 2 * k * (d - d0)
//! ```
//!
//! The force on the central atom is `F = -dV/dd * n_hat` where
//! `n_hat` is the unit normal to the plane.
//!
//! ## Implementation Notes
//!
//! - For atoms I-J-K-L, J is central; plane defined by I, K, L
//! - Force direction is along plane normal, not along position vector
//! - Used for aromatic rings and peptide bonds

use crate::math::Vector;

/// Distance-based improper potential.
///
/// This potential does NOT implement `Potential2` because the force
/// convention is different (force is along plane normal, not along r_vec).
///
/// ## Parameters
///
/// - `k`: Force constant (energy/length² units)
/// - `d0`: Equilibrium out-of-plane distance (length units)
///
/// ## Precomputed Values
///
/// - `two_k`: Stores `2*k` for efficient derivative computation
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dist<T> {
    k: T,
    two_k: T,
    d0: T,
}

impl<T: Vector> Dist<T> {
    /// Creates a new distance-based improper potential.
    ///
    /// ## Arguments
    ///
    /// - `k`: Force constant (energy/length² units)
    /// - `d0`: Equilibrium distance (usually 0)
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::imp::Dist;
    ///
    /// // Planar constraint
    /// let planar = Dist::<f64>::planar(500.0);
    /// ```
    #[inline]
    pub fn new(k: f64, d0: f64) -> Self {
        Self {
            k: T::splat(k),
            two_k: T::splat(2.0 * k),
            d0: T::splat(d0),
        }
    }

    /// Creates for planar geometry (d0 = 0).
    #[inline]
    pub fn planar(k: f64) -> Self {
        Self::new(k, 0.0)
    }

    /// Computes the potential energy.
    ///
    /// ## Arguments
    ///
    /// - `d`: Signed out-of-plane distance
    ///
    /// ## Returns
    ///
    /// ```text
    /// V = k * (d - d0)^2
    /// ```
    #[inline(always)]
    pub fn energy(&self, d: T) -> T {
        let delta = d - self.d0;
        self.k * delta * delta
    }

    /// Computes the force derivative.
    ///
    /// ## Arguments
    ///
    /// - `d`: Signed out-of-plane distance
    ///
    /// ## Returns
    ///
    /// The derivative `dV/dd = 2k * (d - d0)`.
    ///
    /// The force on the central atom is `F = -dV/dd * n_hat` where
    /// `n_hat` is the unit normal to the plane.
    #[inline(always)]
    pub fn derivative(&self, d: T) -> T {
        let delta = d - self.d0;
        self.two_k * delta
    }

    /// Computes energy and derivative together (optimized).
    ///
    /// Shares the computation of `delta`.
    #[inline(always)]
    pub fn energy_derivative(&self, d: T) -> (T, T) {
        let delta = d - self.d0;
        (self.k * delta * delta, self.two_k * delta)
    }
}
