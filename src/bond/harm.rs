//! # Harmonic Bond Potential
//!
//! The standard harmonic spring potential for bond stretching.
//!
//! ## Formula
//!
//! ```text
//! V(r) = k * (r - r0)^2
//! ```
//!
//! where:
//! - `k`: Force constant (energy/length^2 units)
//! - `r0`: Equilibrium bond length (length units)
//!
//! ## Force Factor
//!
//! ```text
//! S = -(dV/dr) / r = 2 * k * (r0/r - 1)
//! ```
//!
//! ## Implementation Notes
//!
//! - Some force fields use `V = (1/2) * k * (r - r0)^2`
//! - To convert: `k_here = k_other / 2`
//! - This is the most common bond potential (AMBER, CHARMM, OPLS)

use crate::base::Potential2;
use crate::math::Vector;

/// Harmonic bond potential.
///
/// ## Parameters
///
/// - `k`: Force constant (energy/length^2 units)
/// - `r0`: Equilibrium distance (length units)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Harm<T> {
    /// Force constant
    k: T,
    /// Equilibrium distance
    r0: T,
}

impl<T: Vector> Harm<T> {
    /// Creates a new harmonic bond potential.
    ///
    /// ## Arguments
    ///
    /// - `k`: Force constant (energy/length^2 units)
    /// - `r0`: Equilibrium bond length (length units)
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::bond::Harm;
    ///
    /// // Typical C-C single bond
    /// let bond = Harm::<f64>::new(300.0, 1.54);
    /// ```
    #[inline]
    pub fn new(k: f64, r0: f64) -> Self {
        Self {
            k: T::splat(k),
            r0: T::splat(r0),
        }
    }

    /// Returns the force constant.
    #[inline]
    pub fn k(&self) -> T {
        self.k
    }

    /// Returns the equilibrium distance.
    #[inline]
    pub fn r0(&self) -> T {
        self.r0
    }
}

impl<T: Vector> Potential2<T> for Harm<T> {
    /// Computes the potential energy.
    ///
    /// ```text
    /// V(r) = k * (r - r0)^2
    /// ```
    #[inline(always)]
    fn energy(&self, r_sq: T) -> T {
        let r = r_sq.sqrt();
        let dr = r - self.r0;
        self.k * dr * dr
    }

    /// Computes the force factor.
    ///
    /// ```text
    /// dV/dr = 2 * k * (r - r0)
    /// S = -(dV/dr)/r = -2 * k * (r - r0) / r
    ///   = 2 * k * (r0/r - 1)
    /// ```
    #[inline(always)]
    fn force_factor(&self, r_sq: T) -> T {
        let r = r_sq.sqrt();
        let two = T::splat(2.0);
        two * self.k * (self.r0 / r - T::one())
    }

    /// Computes energy and force factor together (optimized).
    ///
    /// Shares the computation of `r` and `r_inv`.
    #[inline(always)]
    fn energy_force(&self, r_sq: T) -> (T, T) {
        let r = r_sq.sqrt();
        let r_inv = r.recip();
        let dr = r - self.r0;

        let energy = self.k * dr * dr;

        let two = T::splat(2.0);
        let force = two * self.k * (self.r0 * r_inv - T::one());

        (energy, force)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_harm_at_equilibrium() {
        let k = 300.0;
        let r0 = 1.5;
        let harm: Harm<f64> = Harm::new(k, r0);

        let energy = harm.energy(r0 * r0);
        assert_relative_eq!(energy, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_harm_force_at_equilibrium() {
        let harm: Harm<f64> = Harm::new(300.0, 1.5);
        let r0 = 1.5;

        let force = harm.force_factor(r0 * r0);
        assert_relative_eq!(force, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_harm_stretched() {
        let k = 100.0;
        let r0 = 1.0;
        let harm: Harm<f64> = Harm::new(k, r0);

        let r = 1.5;
        let energy = harm.energy(r * r);

        assert_relative_eq!(energy, 25.0, epsilon = 1e-10);
    }

    #[test]
    fn test_harm_compressed() {
        let k = 100.0;
        let r0 = 1.0;
        let harm: Harm<f64> = Harm::new(k, r0);

        let r = 0.5;
        let energy = harm.energy(r * r);

        assert_relative_eq!(energy, 25.0, epsilon = 1e-10);
    }

    #[test]
    fn test_harm_numerical_derivative() {
        let harm: Harm<f64> = Harm::new(300.0, 1.54);
        let r = 1.6;
        let r_sq = r * r;

        let h = 1e-6;
        let v_plus = harm.energy((r + h) * (r + h));
        let v_minus = harm.energy((r - h) * (r - h));
        let dv_dr_numerical = (v_plus - v_minus) / (2.0 * h);

        let s_numerical = -dv_dr_numerical / r;
        let s_analytical = harm.force_factor(r_sq);

        assert_relative_eq!(s_analytical, s_numerical, epsilon = 1e-6);
    }

    #[test]
    fn test_harm_energy_force_consistency() {
        let harm: Harm<f64> = Harm::new(200.0, 1.2);
        let r_sq = 1.5 * 1.5;

        let (e1, f1) = harm.energy_force(r_sq);
        let e2 = harm.energy(r_sq);
        let f2 = harm.force_factor(r_sq);

        assert_relative_eq!(e1, e2, epsilon = 1e-10);
        assert_relative_eq!(f1, f2, epsilon = 1e-10);
    }
}
