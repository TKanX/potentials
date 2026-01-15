//! # Bond-Bond Cross Term
//!
//! A cross term coupling the two bond lengths adjacent to an angle.
//!
//! ## Formula
//!
//! ```text
//! V(r_ij, r_jk) = k * (r_ij - r_ij0) * (r_jk - r_jk0)
//! ```
//!
//! where:
//! - `k`: Cross coupling constant (energy/length^2 units)
//! - `r_ij0`: Equilibrium length of bond i-j (length units)
//! - `r_jk0`: Equilibrium length of bond j-k (length units)
//!
//! ## Derivative
//!
//! ```text
//! dV/d(cos_theta) = 0  (cross term is angle-independent)
//! ```
//!
//! ## Implementation Notes
//!
//! - Used in Class II force fields (CFF, COMPASS, PCFF)
//! - Captures coupling between adjacent bonds
//! - Main effect is through bond stretch coordinates

use crate::base::Potential3;
use crate::math::Vector;

/// Bond-bond cross term potential.
///
/// ## Parameters
///
/// - `k`: Cross coupling constant (energy/length^2 units)
/// - `r_ij0`: Equilibrium bond length i-j (length units)
/// - `r_jk0`: Equilibrium bond length j-k (length units)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cross<T> {
    /// Cross coupling constant
    k: T,
    /// Equilibrium bond length i-j
    r_ij0: T,
    /// Equilibrium bond length j-k
    r_jk0: T,
}

impl<T: Vector> Cross<T> {
    /// Creates a new bond-bond cross term.
    ///
    /// ## Arguments
    ///
    /// - `k`: Cross coupling constant
    /// - `r_ij0`: Equilibrium bond length i-j
    /// - `r_jk0`: Equilibrium bond length j-k
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::angle::Cross;
    ///
    /// // Cross term for H-C-H angle in methane
    /// let cross = Cross::<f64>::new(20.0, 1.09, 1.09);
    /// ```
    #[inline]
    pub fn new(k: f64, r_ij0: f64, r_jk0: f64) -> Self {
        Self {
            k: T::splat(k),
            r_ij0: T::splat(r_ij0),
            r_jk0: T::splat(r_jk0),
        }
    }

    /// Creates with equal equilibrium distances.
    #[inline]
    pub fn symmetric(k: f64, r0: f64) -> Self {
        Self::new(k, r0, r0)
    }

    /// Returns the cross coupling constant.
    #[inline]
    pub fn k(&self) -> T {
        self.k
    }
}

impl<T: Vector> Potential3<T> for Cross<T> {
    /// Computes the potential energy.
    ///
    /// ```text
    /// V = k * (r_ij - r_ij0) * (r_jk - r_jk0)
    /// ```
    #[inline(always)]
    fn energy(&self, r_ij_sq: T, r_jk_sq: T, _cos_theta: T) -> T {
        let r_ij = r_ij_sq.sqrt();
        let r_jk = r_jk_sq.sqrt();

        let dr_ij = r_ij - self.r_ij0;
        let dr_jk = r_jk - self.r_jk0;

        self.k * dr_ij * dr_jk
    }

    /// Computes dV/d(cos_theta).
    ///
    /// The cross term doesn't depend on angle, so derivative is zero.
    #[inline(always)]
    fn derivative(&self, _r_ij_sq: T, _r_jk_sq: T, _cos_theta: T) -> T {
        T::zero()
    }

    /// Computes energy and derivative together.
    #[inline(always)]
    fn energy_derivative(&self, r_ij_sq: T, r_jk_sq: T, cos_theta: T) -> (T, T) {
        (self.energy(r_ij_sq, r_jk_sq, cos_theta), T::zero())
    }
}
