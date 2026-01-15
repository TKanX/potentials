//! # Morse Bond Potential
//!
//! The Morse potential describes anharmonic bond stretching with
//! correct asymptotic behavior for bond dissociation.
//!
//! ## Formula
//!
//! ```text
//! V(r) = D * (1 - exp(-alpha * (r - r0)))^2
//! ```
//!
//! where:
//! - `D`: Well depth / dissociation energy (energy units)
//! - `alpha`: Width parameter (1/length units)
//! - `r0`: Equilibrium bond length (length units)
//!
//! ## Force Factor
//!
//! ```text
//! S = -(dV/dr) / r = -2 * D * alpha * exp(-alpha*(r-r0)) * (1 - exp(-alpha*(r-r0))) / r
//! ```
//!
//! ## Implementation Notes
//!
//! - Stores `-alpha` internally for efficient `exp(-alpha*dr)` computation
//! - V(r0) = 0, V(∞) = D (dissociation limit)
//! - Near equilibrium: k_eff = 2 * D * alpha^2

use crate::base::Potential2;
use crate::math::Vector;

/// Morse bond potential.
///
/// ## Parameters
///
/// - `d`: Dissociation energy (energy units)
/// - `alpha`: Width parameter (1/length units)
/// - `r0`: Equilibrium distance (length units)
///
/// ## Precomputed Values
///
/// - `neg_alpha`: Stores `-alpha` for efficient `exp(-alpha*dr)` computation
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Morse<T> {
    /// Dissociation energy
    d: T,
    /// Width parameter (stored negative for exp calculation)
    neg_alpha: T,
    /// Equilibrium distance
    r0: T,
}

impl<T: Vector> Morse<T> {
    /// Creates a new Morse potential.
    ///
    /// ## Arguments
    ///
    /// - `d`: Dissociation energy (energy units)
    /// - `alpha`: Width parameter (1/length units)
    /// - `r0`: Equilibrium bond length (length units)
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::bond::Morse;
    ///
    /// // Typical C-H bond
    /// let bond = Morse::<f64>::new(100.0, 2.0, 1.09);
    /// ```
    #[inline]
    pub fn new(d: f64, alpha: f64, r0: f64) -> Self {
        Self {
            d: T::splat(d),
            neg_alpha: T::splat(-alpha),
            r0: T::splat(r0),
        }
    }

    /// Creates a Morse potential from harmonic parameters.
    ///
    /// Uses the relation: `k = 2 * D * alpha^2`
    ///
    /// ## Arguments
    ///
    /// - `k`: Harmonic force constant
    /// - `d`: Dissociation energy
    /// - `r0`: Equilibrium distance
    #[inline]
    pub fn from_harmonic(k: f64, d: f64, r0: f64) -> Self {
        let alpha = (k / (2.0 * d)).sqrt();
        Self::new(d, alpha, r0)
    }

    /// Returns the dissociation energy.
    #[inline]
    pub fn d(&self) -> T {
        self.d
    }

    /// Returns the equilibrium distance.
    #[inline]
    pub fn r0(&self) -> T {
        self.r0
    }
}

impl<T: Vector> Potential2<T> for Morse<T> {
    /// Computes the potential energy.
    ///
    /// ```text
    /// V(r) = D * (1 - exp(-alpha * (r - r0)))^2
    /// ```
    #[inline(always)]
    fn energy(&self, r_sq: T) -> T {
        let r = r_sq.sqrt();
        let dr = r - self.r0;

        // exp(-alpha * dr) = exp(neg_alpha * dr)
        let exp_term = (self.neg_alpha * dr).exp();
        let bracket = T::one() - exp_term;

        self.d * bracket * bracket
    }

    /// Computes the force factor.
    ///
    /// ```text
    /// Let x = exp(-alpha * (r - r0))
    /// V = D * (1 - x)^2
    /// dV/dr = D * 2 * (1 - x) * alpha * x = 2 * D * alpha * x * (1 - x)
    /// S = -(dV/dr)/r = -2 * D * alpha * x * (1 - x) / r
    /// ```
    #[inline(always)]
    fn force_factor(&self, r_sq: T) -> T {
        let r = r_sq.sqrt();
        let r_inv = r.recip();
        let dr = r - self.r0;

        let exp_term = (self.neg_alpha * dr).exp();
        let bracket = T::one() - exp_term;

        // S = -2 * D * (-neg_alpha) * exp_term * bracket / r
        //   = 2 * D * neg_alpha * exp_term * bracket / r (note: neg_alpha is negative)
        let two = T::splat(2.0);
        two * self.d * self.neg_alpha * exp_term * bracket * r_inv
    }

    /// Computes energy and force factor together (optimized).
    ///
    /// Shares the computation of `r`, `r_inv`, `exp_term`, and `bracket`.
    #[inline(always)]
    fn energy_force(&self, r_sq: T) -> (T, T) {
        let r = r_sq.sqrt();
        let r_inv = r.recip();
        let dr = r - self.r0;

        let exp_term = (self.neg_alpha * dr).exp();
        let bracket = T::one() - exp_term;

        let energy = self.d * bracket * bracket;

        let two = T::splat(2.0);
        let force = two * self.d * self.neg_alpha * exp_term * bracket * r_inv;

        (energy, force)
    }
}
