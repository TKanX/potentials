//! # Lennard-Jones 12-6 Potential
//!
//! The standard Lennard-Jones potential for van der Waals interactions.
//!
//! ## Formula
//!
//! ```text
//! V(r) = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]
//!      = epsilon * [(r_min/r)^12 - 2*(r_min/r)^6]
//! ```
//!
//! where:
//! - `epsilon`: Well depth (energy units)
//! - `sigma`: Distance where V(r) = 0 (length units)
//! - `r_min = 2^(1/6) * sigma`: Distance at minimum energy
//!
//! ## Force Factor
//!
//! ```text
//! S = -(dV/dr) / r = 24 * epsilon * [2*(sigma/r)^12 - (sigma/r)^6] / r^2
//! ```
//!
//! ## Implementation Notes
//!
//! - Uses precomputed `sigma^6` and `sigma^12` for efficiency
//! - All operations are branchless (no conditionals)
//! - Accepts `r^2` as input to avoid square root in distance calculation

use crate::base::Potential2;
use crate::math::Vector;

/// Lennard-Jones 12-6 potential.
///
/// ## Parameters
///
/// - `epsilon`: Well depth (energy)
/// - `sigma`: Zero-crossing distance (length)
///
/// ## Precomputed Values
///
/// Internally stores:
/// - `c6 = 4 * epsilon * sigma^6`
/// - `c12 = 4 * epsilon * sigma^12`
///
/// This allows the potential to be computed as: `V = c12/r^12 - c6/r^6`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lj<T> {
    /// Coefficient for r^-12 term: 4 * epsilon * sigma^12
    c12: T,
    /// Coefficient for r^-6 term: 4 * epsilon * sigma^6
    c6: T,
}

impl<T: Vector> Lj<T> {
    /// Creates a new Lennard-Jones potential from epsilon and sigma.
    ///
    /// ## Arguments
    ///
    /// - `epsilon`: Well depth (energy units)
    /// - `sigma`: Zero-crossing distance (length units)
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::pair::Lj;
    ///
    /// // Argon: epsilon = 0.238 kcal/mol, sigma = 3.4 A
    /// let lj: Lj<f64> = Lj::new(0.238, 3.4);
    /// ```
    #[inline]
    pub fn new(epsilon: f64, sigma: f64) -> Self {
        let sigma_sq = sigma * sigma;
        let sigma_6 = sigma_sq * sigma_sq * sigma_sq;
        let sigma_12 = sigma_6 * sigma_6;
        let four_eps = 4.0 * epsilon;

        Self {
            c12: T::splat(four_eps * sigma_12),
            c6: T::splat(four_eps * sigma_6),
        }
    }

    /// Creates a new Lennard-Jones potential from precomputed coefficients.
    ///
    /// ## Arguments
    ///
    /// - `c12`: Coefficient for r^-12 term (4 * epsilon * sigma^12)
    /// - `c6`: Coefficient for r^-6 term (4 * epsilon * sigma^6)
    ///
    /// This is useful when mixing rules produce c6/c12 directly.
    #[inline]
    pub fn from_coefficients(c12: f64, c6: f64) -> Self {
        Self {
            c12: T::splat(c12),
            c6: T::splat(c6),
        }
    }

    /// Returns the c12 coefficient.
    #[inline]
    pub fn c12(&self) -> T {
        self.c12
    }

    /// Returns the c6 coefficient.
    #[inline]
    pub fn c6(&self) -> T {
        self.c6
    }
}

impl<T: Vector> Potential2<T> for Lj<T> {
    /// Computes the potential energy.
    ///
    /// ```text
    /// V(r) = c12/r^12 - c6/r^6
    /// ```
    #[inline(always)]
    fn energy(&self, r_sq: T) -> T {
        // Compute r^-6 efficiently
        let r_sq_inv = r_sq.recip(); // 1/r^2
        let r6_inv = r_sq_inv * r_sq_inv * r_sq_inv; // 1/r^6
        let r12_inv = r6_inv * r6_inv; // 1/r^12

        // V = c12/r^12 - c6/r^6
        self.c12 * r12_inv - self.c6 * r6_inv
    }

    /// Computes the force factor.
    ///
    /// ```text
    /// dV/dr = -12*c12/r^13 + 6*c6/r^7
    /// S = -(dV/dr)/r = 12*c12/r^14 - 6*c6/r^8
    ///   = 6 * (2*c12/r^12 - c6/r^6) / r^2
    /// ```
    #[inline(always)]
    fn force_factor(&self, r_sq: T) -> T {
        let r_sq_inv = r_sq.recip();
        let r6_inv = r_sq_inv * r_sq_inv * r_sq_inv;
        let r12_inv = r6_inv * r6_inv;

        // S = 12*c12/r^14 - 6*c6/r^8 = (12*c12*r^-12 - 6*c6*r^-6) / r^2
        let six = T::splat(6.0);
        let twelve = T::splat(12.0);
        (twelve * self.c12 * r12_inv - six * self.c6 * r6_inv) * r_sq_inv
    }

    /// Computes energy and force factor together (optimized).
    ///
    /// Shares the computation of r^-6 and r^-12.
    #[inline(always)]
    fn energy_force(&self, r_sq: T) -> (T, T) {
        let r_sq_inv = r_sq.recip();
        let r6_inv = r_sq_inv * r_sq_inv * r_sq_inv;
        let r12_inv = r6_inv * r6_inv;

        // Precompute common terms
        let c12_r12 = self.c12 * r12_inv;
        let c6_r6 = self.c6 * r6_inv;

        // Energy: c12/r^12 - c6/r^6
        let energy = c12_r12 - c6_r6;

        // Force factor: (12*c12/r^12 - 6*c6/r^6) / r^2
        let six = T::splat(6.0);
        let twelve = T::splat(12.0);
        let force = (twelve * c12_r12 - six * c6_r6) * r_sq_inv;

        (energy, force)
    }
}
