//! # Softcore Potential Wrapper
//!
//! Removes the singularity at r=0 for use in Free Energy Perturbation (FEP)
//! and thermodynamic integration calculations.
//!
//! ## Formula
//!
//! ```text
//! r_eff^2 = alpha * sigma^2 * lambda^n + r^2
//!
//! V_sc(r) = lambda^m * V(r_eff)
//! ```
//!
//! Common choices:
//! - alpha = 0.5
//! - n = 1, m = 1 (linear coupling)
//! - sigma = Lennard-Jones sigma or typical interaction range
//!
//! ## Properties
//!
//! - At lambda=0: r_eff = sqrt(alpha)*sigma, potential decoupled
//! - At lambda=1: r_eff = r, original potential recovered
//! - Finite energy at r=0 for all lambda > 0
//!
//! ## References
//!
//! Beutler et al., Chem. Phys. Lett. 222, 529 (1994)

use crate::base::Potential2;
use crate::math::Vector;

/// Softcore potential for FEP calculations.
///
/// ## Type Parameters
///
/// - `P`: The underlying potential type
/// - `T`: The vector type
#[derive(Clone, Copy, Debug)]
pub struct Softcore<P, T> {
    inner: P,
    alpha_sigma_sq: T, // alpha * sigma^2
    lambda: T,
    lambda_n: T, // lambda^n
    lambda_m: T, // lambda^m
}

impl<P, T: Vector> Softcore<P, T> {
    /// Creates a new softcore potential.
    ///
    /// ## Arguments
    ///
    /// - `inner`: The potential to modify
    /// - `alpha`: Softcore parameter (typically 0.5)
    /// - `sigma`: Length scale for softcore (typically LJ sigma) (length)
    /// - `lambda`: Coupling parameter (0 = decoupled, 1 = coupled)
    /// - `n`: Exponent for distance modification (typically 1)
    /// - `m`: Exponent for energy scaling (typically 1)
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::{pair::Lj, meta::Softcore};
    ///
    /// let lj = Lj::<f64>::new(1.0, 3.4);
    /// let lj_soft: Softcore<_, f64> = Softcore::new(lj, 0.5, 3.4, 0.5, 1, 1);
    /// ```
    #[inline]
    pub fn new(inner: P, alpha: f64, sigma: f64, lambda: f64, n: u32, m: u32) -> Self {
        let lambda_n = lambda.powi(n as i32);
        let lambda_m = lambda.powi(m as i32);

        Self {
            inner,
            alpha_sigma_sq: T::splat(alpha * sigma * sigma),
            lambda: T::splat(lambda),
            lambda_n: T::splat(lambda_n),
            lambda_m: T::splat(lambda_m),
        }
    }

    /// Updates the lambda value.
    ///
    /// Use this during FEP simulations to change the coupling strength.
    #[inline]
    pub fn set_lambda(&mut self, lambda: f64, n: u32, m: u32) {
        self.lambda = T::splat(lambda);
        self.lambda_n = T::splat(lambda.powi(n as i32));
        self.lambda_m = T::splat(lambda.powi(m as i32));
    }

    /// Computes the effective r^2 for softcore interaction.
    #[inline(always)]
    fn effective_r_sq(&self, r_sq: T) -> T {
        self.alpha_sigma_sq * self.lambda_n + r_sq
    }
}

impl<P: Potential2<T>, T: Vector> Potential2<T> for Softcore<P, T> {
    #[inline(always)]
    fn energy(&self, r_sq: T) -> T {
        let r_eff_sq = self.effective_r_sq(r_sq);
        self.lambda_m * self.inner.energy(r_eff_sq)
    }

    #[inline(always)]
    fn force_factor(&self, r_sq: T) -> T {
        let r_eff_sq = self.effective_r_sq(r_sq);
        self.lambda_m * self.inner.force_factor(r_eff_sq)
    }

    #[inline(always)]
    fn energy_force(&self, r_sq: T) -> (T, T) {
        let r_eff_sq = self.effective_r_sq(r_sq);
        let (e, f) = self.inner.energy_force(r_eff_sq);
        (self.lambda_m * e, self.lambda_m * f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pair::Lj;
    use approx::assert_relative_eq;

    #[test]
    fn test_softcore_lambda_one() {
        let lj: Lj<f64> = Lj::new(1.0, 3.4);
        let lj_soft = Softcore::new(lj, 0.5, 3.4, 1.0, 1, 1);

        let r_sq = 16.0;

        let (e, f) = lj_soft.energy_force(r_sq);
        let r_eff_sq = 0.5 * 3.4 * 3.4 + r_sq;
        let (e_expected, f_expected) = lj.energy_force(r_eff_sq);

        assert_relative_eq!(e, e_expected, epsilon = 1e-10);
        assert_relative_eq!(f, f_expected, epsilon = 1e-10);
    }

    #[test]
    fn test_softcore_lambda_zero() {
        let lj: Lj<f64> = Lj::new(1.0, 3.4);
        let lj_soft = Softcore::new(lj, 0.5, 3.4, 0.0, 1, 1);

        let r_sq = 16.0;
        let e = lj_soft.energy(r_sq);

        assert_relative_eq!(e, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_softcore_finite_at_zero() {
        let lj: Lj<f64> = Lj::new(1.0, 3.4);
        let lj_soft = Softcore::new(lj, 0.5, 3.4, 0.5, 1, 1);

        let e = lj_soft.energy(0.0);

        assert!(e.is_finite());
        assert!(e.abs() < 1e10);
    }

    #[test]
    fn test_softcore_numerical_derivative() {
        let lj: Lj<f64> = Lj::new(1.0, 3.4);
        let lj_soft = Softcore::new(lj, 0.5, 3.4, 0.5, 1, 1);

        let r = 4.0;

        let h = 1e-7;
        let e_plus = lj_soft.energy((r + h) * (r + h));
        let e_minus = lj_soft.energy((r - h) * (r - h));
        let dv_dr_numerical = (e_plus - e_minus) / (2.0 * h);

        let f = lj_soft.force_factor(r * r);
        let dv_dr_analytical = -f * r;

        assert_relative_eq!(dv_dr_analytical, dv_dr_numerical, epsilon = 1e-6);
    }
}
