//! # Sum of Potentials
//!
//! Combines two potentials by adding their energies and forces.
//!
//! ## Formula
//!
//! ```text
//! V_sum(r) = V_a(r) + V_b(r)
//! F_sum(r) = F_a(r) + F_b(r)
//! ```
//!
//! ## Applications
//!
//! - Combining LJ with electrostatics
//! - Adding restraints to base potential
//! - Building complex potentials from primitives
//!
//! ## Notes
//!
//! For more than two potentials, nest Sum types:
//! ```text
//! Sum<Sum<A, B>, C> for A + B + C
//! ```

use crate::base::Potential2;
use crate::math::Vector;

/// Sum of two potentials.
///
/// ## Type Parameters
///
/// - `A`: First potential type
/// - `B`: Second potential type
/// - `T`: The vector type
#[derive(Clone, Copy, Debug)]
pub struct Sum<A, B, T> {
    a: A,
    b: B,
    _marker: core::marker::PhantomData<T>,
}

impl<A, B, T> Sum<A, B, T> {
    /// Creates a new sum of two potentials.
    ///
    /// ## Arguments
    ///
    /// - `a`: First potential
    /// - `b`: Second potential
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::{pair::{Lj, Coul}, meta::Sum};
    ///
    /// let lj = Lj::<f64>::new(1.0, 3.4);
    /// let coul = Coul::<f64>::new(1.0); // k * q1 * q2
    /// let total: Sum<_, _, f64> = Sum::new(lj, coul);
    /// ```
    #[inline]
    pub fn new(a: A, b: B) -> Self {
        Self {
            a,
            b,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<A: Potential2<T>, B: Potential2<T>, T: Vector> Potential2<T> for Sum<A, B, T> {
    #[inline(always)]
    fn energy(&self, r_sq: T) -> T {
        self.a.energy(r_sq) + self.b.energy(r_sq)
    }

    #[inline(always)]
    fn force_factor(&self, r_sq: T) -> T {
        self.a.force_factor(r_sq) + self.b.force_factor(r_sq)
    }

    #[inline(always)]
    fn energy_force(&self, r_sq: T) -> (T, T) {
        let (ea, fa) = self.a.energy_force(r_sq);
        let (eb, fb) = self.b.energy_force(r_sq);
        (ea + eb, fa + fb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pair::{Gauss, Lj};
    use approx::assert_relative_eq;

    #[test]
    fn test_sum_energy() {
        let lj: Lj<f64> = Lj::new(1.0, 3.4);
        let gauss: Gauss<f64> = Gauss::new(1.0, 0.5);
        let sum = Sum::new(lj, gauss);

        let r_sq = 16.0;
        let e_lj = lj.energy(r_sq);
        let e_gauss = gauss.energy(r_sq);
        let e_sum = sum.energy(r_sq);

        assert_relative_eq!(e_sum, e_lj + e_gauss, epsilon = 1e-10);
    }

    #[test]
    fn test_sum_force() {
        let lj: Lj<f64> = Lj::new(1.0, 3.4);
        let gauss: Gauss<f64> = Gauss::new(1.0, 0.5);
        let sum = Sum::new(lj, gauss);

        let r_sq = 16.0;
        let f_lj = lj.force_factor(r_sq);
        let f_gauss = gauss.force_factor(r_sq);
        let f_sum = sum.force_factor(r_sq);

        assert_relative_eq!(f_sum, f_lj + f_gauss, epsilon = 1e-10);
    }

    #[test]
    fn test_sum_energy_force() {
        let lj: Lj<f64> = Lj::new(1.0, 3.4);
        let gauss: Gauss<f64> = Gauss::new(1.0, 0.5);
        let sum = Sum::new(lj, gauss);

        let r_sq = 16.0;
        let (e_lj, f_lj) = lj.energy_force(r_sq);
        let (e_gauss, f_gauss) = gauss.energy_force(r_sq);
        let (e_sum, f_sum) = sum.energy_force(r_sq);

        assert_relative_eq!(e_sum, e_lj + e_gauss, epsilon = 1e-10);
        assert_relative_eq!(f_sum, f_lj + f_gauss, epsilon = 1e-10);
    }

    #[test]
    fn test_nested_sum() {
        let p1: Lj<f64> = Lj::new(1.0, 3.4);
        let p2: Gauss<f64> = Gauss::new(1.0, 0.5);
        let p3: Gauss<f64> = Gauss::new(0.5, 0.3);

        let sum12 = Sum::new(p1, p2);
        let sum123 = Sum::new(sum12, p3);

        let r_sq = 16.0;
        let e1 = p1.energy(r_sq);
        let e2 = p2.energy(r_sq);
        let e3 = p3.energy(r_sq);
        let e_total = sum123.energy(r_sq);

        assert_relative_eq!(e_total, e1 + e2 + e3, epsilon = 1e-10);
    }
}
