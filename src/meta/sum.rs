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
