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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pair::Lj;
    use approx::assert_relative_eq;

    #[test]
    fn test_scaled_identity() {
        let lj: Lj<f64> = Lj::new(1.0, 3.4);
        let lj_scaled = Scaled::new(lj, 1.0);

        let r_sq = 16.0;
        let (e, f) = lj.energy_force(r_sq);
        let (e_scaled, f_scaled) = lj_scaled.energy_force(r_sq);

        assert_relative_eq!(e, e_scaled, epsilon = 1e-10);
        assert_relative_eq!(f, f_scaled, epsilon = 1e-10);
    }

    #[test]
    fn test_scaled_zero() {
        let lj: Lj<f64> = Lj::new(1.0, 3.4);
        let lj_scaled = Scaled::new(lj, 0.0);

        let r_sq = 16.0;
        let (e, f) = lj_scaled.energy_force(r_sq);

        assert_relative_eq!(e, 0.0, epsilon = 1e-10);
        assert_relative_eq!(f, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_scaled_half() {
        let lj: Lj<f64> = Lj::new(1.0, 3.4);
        let lj_scaled = Scaled::new(lj, 0.5);

        let r_sq = 16.0;
        let (e_base, f_base) = lj.energy_force(r_sq);
        let (e_scaled, f_scaled) = lj_scaled.energy_force(r_sq);

        assert_relative_eq!(e_scaled, 0.5 * e_base, epsilon = 1e-10);
        assert_relative_eq!(f_scaled, 0.5 * f_base, epsilon = 1e-10);
    }

    #[test]
    fn test_scaled_negative() {
        let lj: Lj<f64> = Lj::new(1.0, 3.4);
        let lj_scaled = Scaled::new(lj, -1.0);

        let r_sq = 16.0;
        let (e_base, f_base) = lj.energy_force(r_sq);
        let (e_scaled, f_scaled) = lj_scaled.energy_force(r_sq);

        assert_relative_eq!(e_scaled, -e_base, epsilon = 1e-10);
        assert_relative_eq!(f_scaled, -f_base, epsilon = 1e-10);
    }
}
