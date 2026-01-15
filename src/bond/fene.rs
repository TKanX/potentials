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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fene_at_zero() {
        let fene: Fene<f64> = Fene::new(30.0, 1.5);

        let energy = fene.energy(1e-20);
        assert_relative_eq!(energy, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fene_increases() {
        let fene: Fene<f64> = Fene::new(30.0, 1.5);

        let e1 = fene.energy(0.5);
        let e2 = fene.energy(1.0);
        let e3 = fene.energy(1.5);

        assert!(e1 < e2, "Energy should increase");
        assert!(e2 < e3, "Energy should increase");
    }

    #[test]
    fn test_fene_harmonic_limit() {
        let k = 30.0;
        let r_max = 1.5;
        let fene: Fene<f64> = Fene::new(k, r_max);

        let r = 0.1;
        let r_sq = r * r;

        let v_fene = fene.energy(r_sq);
        let v_harmonic = 0.5 * k * r_sq;

        let rel_diff = (v_fene - v_harmonic).abs() / v_harmonic;
        assert!(
            rel_diff < 0.01,
            "FENE {} vs harmonic {}, diff {}%",
            v_fene,
            v_harmonic,
            rel_diff * 100.0
        );
    }

    #[test]
    fn test_fene_numerical_derivative() {
        let fene: Fene<f64> = Fene::new(30.0, 1.5);
        let r = 0.8;
        let r_sq = r * r;

        let h = 1e-6;
        let v_plus = fene.energy((r + h) * (r + h));
        let v_minus = fene.energy((r - h) * (r - h));
        let dv_dr_numerical = (v_plus - v_minus) / (2.0 * h);

        let s_numerical = -dv_dr_numerical / r;
        let s_analytical = fene.force_factor(r_sq);

        assert_relative_eq!(s_analytical, s_numerical, epsilon = 1e-6);
    }
}
