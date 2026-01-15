//! # Scaled Potential Wrapper
//!
//! Linearly scales potential energy and forces by a constant factor.
//!
//! ## Formula
//!
//! ```text
//! V_scaled(r) = scale * V(r)
//! F_scaled(r) = scale * F(r)
//! ```
//!
//! ## Notes
//!
//! - Free Energy Perturbation (simpler than softcore)
//! - Weighted potential combinations
//! - Coupling/decoupling interactions gradually

use crate::base::Potential2;
use crate::math::Vector;

/// Linearly scaled potential.
///
/// ## Type Parameters
///
/// - `P`: The underlying potential type
/// - `T`: The vector type
#[derive(Clone, Copy, Debug)]
pub struct Scaled<P, T> {
    inner: P,
    scale: T,
}

impl<P, T: Vector> Scaled<P, T> {
    /// Creates a new scaled potential.
    ///
    /// ## Arguments
    ///
    /// - `inner`: The potential to scale
    /// - `scale`: Scale factor (1.0 = no change, 0.0 = disabled)
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::{pair::Lj, meta::Scaled};
    ///
    /// let lj = Lj::<f64>::new(1.0, 3.4);
    /// let lj_half: Scaled<_, f64> = Scaled::new(lj, 0.5);
    /// ```
    #[inline]
    pub fn new(inner: P, scale: f64) -> Self {
        Self {
            inner,
            scale: T::splat(scale),
        }
    }

    /// Updates the scale factor.
    #[inline]
    pub fn set_scale(&mut self, scale: f64) {
        self.scale = T::splat(scale);
    }
}

impl<P: Potential2<T>, T: Vector> Potential2<T> for Scaled<P, T> {
    #[inline(always)]
    fn energy(&self, r_sq: T) -> T {
        self.scale * self.inner.energy(r_sq)
    }

    #[inline(always)]
    fn force_factor(&self, r_sq: T) -> T {
        self.scale * self.inner.force_factor(r_sq)
    }

    #[inline(always)]
    fn energy_force(&self, r_sq: T) -> (T, T) {
        let (e, f) = self.inner.energy_force(r_sq);
        (self.scale * e, self.scale * f)
    }
}
