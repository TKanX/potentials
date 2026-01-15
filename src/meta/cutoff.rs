//! # Hard Cutoff Wrapper
//!
//! Sets potential to zero beyond a cutoff distance.
//!
//! ## Formula
//!
//! ```text
//! V_cut(r) = V(r) if r < rc
//!          = 0    if r >= rc
//! ```
//!
//! ## Notes
//!
//! - Simple truncation (discontinuous at rc)
//! - For smooth cutoffs, use [`Switch`] or [`Shift`]
//! - Implemented branchlessly using mask operations

use crate::base::Potential2;
use crate::math::{Mask, Vector};

/// Hard cutoff wrapper.
///
/// ## Type Parameters
///
/// - `P`: The underlying potential type
/// - `T`: The vector type
#[derive(Clone, Copy, Debug)]
pub struct Cutoff<P, T> {
    inner: P,
    rc_sq: T,
}

impl<P, T: Vector> Cutoff<P, T> {
    /// Creates a new cutoff wrapper.
    ///
    /// ## Arguments
    ///
    /// - `inner`: The potential to wrap
    /// - `rc`: Cutoff distance (length)
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::{pair::Lj, meta::Cutoff};
    ///
    /// let lj = Lj::<f64>::new(1.0, 3.4);
    /// let lj_cut: Cutoff<_, f64> = Cutoff::new(lj, 12.0);
    /// ```
    #[inline]
    pub fn new(inner: P, rc: f64) -> Self {
        Self {
            inner,
            rc_sq: T::splat(rc * rc),
        }
    }

    /// Returns a reference to the inner potential.
    #[inline]
    pub fn inner(&self) -> &P {
        &self.inner
    }
}

impl<P: Potential2<T>, T: Vector> Potential2<T> for Cutoff<P, T> {
    #[inline(always)]
    fn energy(&self, r_sq: T) -> T {
        let e = self.inner.energy(r_sq);
        let mask = r_sq.lt(self.rc_sq);
        mask.select(e, T::zero())
    }

    #[inline(always)]
    fn force_factor(&self, r_sq: T) -> T {
        let s = self.inner.force_factor(r_sq);
        let mask = r_sq.lt(self.rc_sq);
        mask.select(s, T::zero())
    }

    #[inline(always)]
    fn energy_force(&self, r_sq: T) -> (T, T) {
        let (e, s) = self.inner.energy_force(r_sq);
        let mask = r_sq.lt(self.rc_sq);
        (mask.select(e, T::zero()), mask.select(s, T::zero()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pair::Lj;
    use approx::assert_relative_eq;

    #[test]
    fn test_cutoff_inside() {
        let lj: Lj<f64> = Lj::new(1.0, 3.4);
        let lj_cut = Cutoff::new(lj, 12.0);

        let r_sq = 16.0;

        let e_base = lj.energy(r_sq);
        let e_cut = lj_cut.energy(r_sq);

        assert_relative_eq!(e_base, e_cut, epsilon = 1e-10);
    }

    #[test]
    fn test_cutoff_outside() {
        let lj: Lj<f64> = Lj::new(1.0, 3.4);
        let lj_cut = Cutoff::new(lj, 10.0);

        let r_sq = 121.0;

        let e = lj_cut.energy(r_sq);
        assert_relative_eq!(e, 0.0, epsilon = 1e-10);

        let s = lj_cut.force_factor(r_sq);
        assert_relative_eq!(s, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cutoff_at_boundary() {
        let lj: Lj<f64> = Lj::new(1.0, 3.4);
        let rc = 10.0;
        let lj_cut = Cutoff::new(lj, rc);

        let r_in = rc - 0.001;
        let e_in = lj_cut.energy(r_in * r_in);
        assert!(e_in != 0.0);

        let r_out = rc + 0.001;
        let e_out = lj_cut.energy(r_out * r_out);
        assert_relative_eq!(e_out, 0.0, epsilon = 1e-10);
    }
}
