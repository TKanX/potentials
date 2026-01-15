//! # FENE Bond Potential
//!
//! The Finitely Extensible Nonlinear Elastic (FENE) potential for
//! polymer chain connectivity with a maximum extension limit.
//!
//! ## Formula
//!
//! ```text
//! V(r) = -(k/2) * R^2 * ln(1 - (r/R)^2)    for r < R
//!      = infinity                          for r >= R
//! ```
//!
//! where:
//! - `k`: Spring constant (energy/length^2 units)
//! - `R`: Maximum extension (length units)
//!
//! ## Force Factor
//!
//! ```text
//! S = -(dV/dr) / r = -k * R^2 / (R^2 - r^2)
//! ```
//!
//! ## Implementation Notes
//!
//! - Finite extensibility: r cannot exceed R
//! - V(0) = 0, V(r) -> ∞ as r -> R
//! - Near r=0: V ≈ (k/2) * r^2 (harmonic)
//! - Often combined with WCA for polymer simulations

use crate::base::Potential2;
use crate::math::Vector;

/// FENE bond potential.
///
/// ## Parameters
///
/// - `k`: Spring constant (energy/length^2 units)
/// - `r_max`: Maximum extension (length units)
///
/// ## Precomputed Values
///
/// - `prefactor`: Stores `-k/2 * R^2` for efficient computation
/// - `inv_r_max_sq`: Stores `1/R^2` for branchless calculation
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fene<T> {
    /// Prefactor: -k/2 * R^2
    prefactor: T,
    /// Maximum extension squared (R^2)
    r_max_sq: T,
    /// 1 / R^2 for branchless calculation
    inv_r_max_sq: T,
}

impl<T: Vector> Fene<T> {
    /// Creates a new FENE potential.
    ///
    /// ## Arguments
    ///
    /// - `k`: Spring constant (energy/length^2 units)
    /// - `r_max`: Maximum extension (length units)
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::bond::Fene;
    ///
    /// // Standard FENE for polymer chains
    /// // Often paired with k=30 epsilon/sigma^2, R=1.5 sigma
    /// let bond = Fene::<f64>::new(30.0, 1.5);
    /// ```
    #[inline]
    pub fn new(k: f64, r_max: f64) -> Self {
        let r_max_sq = r_max * r_max;
        Self {
            prefactor: T::splat(-0.5 * k * r_max_sq),
            r_max_sq: T::splat(r_max_sq),
            inv_r_max_sq: T::splat(1.0 / r_max_sq),
        }
    }

    /// Returns the maximum extension squared.
    #[inline]
    pub fn r_max_sq(&self) -> T {
        self.r_max_sq
    }
}

impl<T: Vector> Potential2<T> for Fene<T> {
    /// Computes the potential energy.
    ///
    /// ```text
    /// V(r) = -(k/2) * R^2 * ln(1 - (r/R)^2)
    ///      = prefactor * ln(1 - r^2/R^2)
    /// ```
    ///
    /// ## Warning
    ///
    /// Returns large positive value (not infinity) for r >= R.
    /// The caller should ensure r < R.
    #[inline(always)]
    fn energy(&self, r_sq: T) -> T {
        // ratio = r^2 / R^2
        let ratio = r_sq * self.inv_r_max_sq;

        // ln(1 - ratio)
        // Note: For r >= R, this will be ln of negative, returning NaN
        // We use branchless math: clamp ratio to avoid NaN
        let one = T::one();
        let safe_ratio = ratio.min(T::splat(0.9999)); // Prevent ln(0) or ln(negative)
        let ln_arg = one - safe_ratio;

        self.prefactor * ln_arg.ln()
    }

    /// Computes the force factor.
    ///
    /// ```text
    /// V = prefactor * ln(1 - r^2/R^2)
    /// dV/dr = prefactor * (-2r/R^2) / (1 - r^2/R^2)
    ///       = -2 * prefactor * r / (R^2 - r^2)
    /// S = -(dV/dr)/r = 2 * prefactor / (R^2 - r^2)
    ///   = -k * R^2 / (R^2 - r^2)
    /// ```
    #[inline(always)]
    fn force_factor(&self, r_sq: T) -> T {
        // R^2 - r^2
        let denom = self.r_max_sq - r_sq;

        // Prevent division by zero for r >= R
        let safe_denom = denom.max(T::splat(1e-10));

        // S = 2 * prefactor / denom (prefactor is negative)
        let two = T::splat(2.0);
        two * self.prefactor / safe_denom
    }

    /// Computes energy and force factor together (optimized).
    ///
    /// Shares the computation of `ratio` and `ln_arg`.
    #[inline(always)]
    fn energy_force(&self, r_sq: T) -> (T, T) {
        let ratio = r_sq * self.inv_r_max_sq;
        let one = T::one();

        let safe_ratio = ratio.min(T::splat(0.9999));
        let ln_arg = one - safe_ratio;

        let energy = self.prefactor * ln_arg.ln();

        // R^2 - r^2 = R^2 * (1 - r^2/R^2) = R^2 * ln_arg
        let denom = self.r_max_sq * ln_arg;
        let safe_denom = denom.max(T::splat(1e-10));

        let two = T::splat(2.0);
        let force = two * self.prefactor / safe_denom;

        (energy, force)
    }
}
