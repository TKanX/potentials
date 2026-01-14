//! # Buckingham Potential (Exp-6)
//!
//! The Buckingham potential models short-range repulsion with an exponential
//! term and long-range attraction with an r^-6 dispersion term.
//!
//! ## Formula
//!
//! ```text
//! V(r) = A * exp(-B * r) - C / r^6
//! ```
//!
//! where:
//! - `A`: Repulsion amplitude (energy units)
//! - `B`: Repulsion steepness (1/length units)
//! - `C`: Dispersion coefficient (energy * length^6 units)
//!
//! ## Force Factor
//!
//! ```text
//! S = -(dV/dr) / r = A * B * exp(-B * r) / r - 6 * C / r^8
//! ```
//!
//! ## Implementation Notes
//!
//! - Stores `-B` internally for efficient `exp(-B*r)` computation
//! - Caution: Goes to -infinity as r -> 0 ("Buckingham catastrophe")
//! - Always use with a short-range cutoff or repulsive wall at small r

use crate::base::Potential2;
use crate::math::Vector;

/// Buckingham (Exp-6) potential.
///
/// ## Parameters
///
/// - `a`: Repulsion amplitude (energy units)
/// - `b`: Repulsion steepness (1/length units)
/// - `c`: Dispersion coefficient (energy * length^6 units)
///
/// ## Precomputed Values
///
/// - `neg_b`: Stores `-b` internally for efficient `exp(-b*r)` computation
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Buck<T> {
    /// Repulsion amplitude
    a: T,
    /// Repulsion decay rate (negative for efficiency: stores -B)
    neg_b: T,
    /// Dispersion coefficient
    c: T,
}

impl<T: Vector> Buck<T> {
    /// Creates a new Buckingham potential.
    ///
    /// ## Arguments
    ///
    /// - `a`: Repulsion amplitude (energy units)
    /// - `b`: Repulsion steepness (1/length units)
    /// - `c`: Dispersion coefficient (energy * length^6 units)
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::pair::Buck;
    ///
    /// // NaCl parameters (example values)
    /// let buck = Buck::<f64>::new(1000.0, 3.0, 100.0);
    /// ```
    #[inline]
    pub fn new(a: f64, b: f64, c: f64) -> Self {
        Self {
            a: T::splat(a),
            neg_b: T::splat(-b),
            c: T::splat(c),
        }
    }

    /// Creates a Buckingham potential from rho parameterization.
    ///
    /// ```text
    /// V(r) = A * exp(-r / rho) - C / r^6
    /// ```
    ///
    /// ## Arguments
    ///
    /// - `a`: Repulsion amplitude
    /// - `rho`: Softness parameter (rho = 1/b)
    /// - `c`: Dispersion coefficient
    #[inline]
    pub fn from_rho(a: f64, rho: f64, c: f64) -> Self {
        Self::new(a, 1.0 / rho, c)
    }

    /// Returns the A parameter.
    #[inline]
    pub fn a(&self) -> T {
        self.a
    }

    /// Returns the C parameter.
    #[inline]
    pub fn c(&self) -> T {
        self.c
    }
}

impl<T: Vector> Potential2<T> for Buck<T> {
    /// Computes the potential energy.
    ///
    /// ```text
    /// V(r) = A * exp(-B * r) - C / r^6
    /// ```
    #[inline(always)]
    fn energy(&self, r_sq: T) -> T {
        let r = r_sq.sqrt();
        let r_sq_inv = r_sq.recip();
        let r6_inv = r_sq_inv * r_sq_inv * r_sq_inv;

        // exp(-B*r) = exp(neg_b * r)
        let exp_term = (self.neg_b * r).exp();

        self.a * exp_term - self.c * r6_inv
    }

    /// Computes the force factor.
    ///
    /// ```text
    /// dV/dr = -A*B*exp(-B*r) + 6*C/r^7
    /// S = -(dV/dr)/r = A*B*exp(-B*r)/r - 6*C/r^8
    /// ```
    #[inline(always)]
    fn force_factor(&self, r_sq: T) -> T {
        let r = r_sq.sqrt();
        let r_inv = r.recip();
        let r_sq_inv = r_sq.recip();
        let r6_inv = r_sq_inv * r_sq_inv * r_sq_inv;

        let exp_term = (self.neg_b * r).exp();
        let six = T::splat(6.0);

        // S = -neg_b * A * exp(-B*r) / r - 6*C/r^8
        // Note: neg_b = -B, so -neg_b = B
        T::zero() - self.neg_b * self.a * exp_term * r_inv - six * self.c * r6_inv * r_sq_inv
    }

    /// Computes energy and force factor together (optimized).
    ///
    /// Shares the computation of `r`, `r_inv`, `r_sq_inv`, `r6_inv`, and `exp_term`.
    #[inline(always)]
    fn energy_force(&self, r_sq: T) -> (T, T) {
        let r = r_sq.sqrt();
        let r_inv = r.recip();
        let r_sq_inv = r_sq.recip();
        let r6_inv = r_sq_inv * r_sq_inv * r_sq_inv;

        let exp_term = (self.neg_b * r).exp();
        let a_exp = self.a * exp_term;
        let c_r6 = self.c * r6_inv;

        let energy = a_exp - c_r6;

        let six = T::splat(6.0);
        let force = T::zero() - self.neg_b * a_exp * r_inv - six * c_r6 * r_sq_inv;

        (energy, force)
    }
}
