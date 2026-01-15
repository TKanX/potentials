//! # Shifted Potential Wrapper
//!
//! Shifts potential so it's exactly zero at the cutoff distance.
//!
//! ## Formula
//!
//! ```text
//! V_shift(r) = V(r) - V(rc)  if r < rc
//!            = 0              if r >= rc
//! ```
//!
//! ## Notes
//!
//! - Ensures energy continuity at cutoff (no jump)
//! - Force still discontinuous (use Switch for smooth force)
//! - Very common in molecular simulations

use crate::base::Potential2;
use crate::math::{Mask, Vector};

/// Shifted potential wrapper.
///
/// ## Type Parameters
///
/// - `P`: The underlying potential type
/// - `T`: The vector type
#[derive(Clone, Copy, Debug)]
pub struct Shift<P, T> {
    inner: P,
    rc_sq: T,
    e_rc: T, // Energy at cutoff
}

impl<P: Potential2<T>, T: Vector> Shift<P, T> {
    /// Creates a new shifted potential.
    ///
    /// ## Arguments
    ///
    /// - `inner`: The potential to wrap
    /// - `rc`: Cutoff distance (length)
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::{pair::Lj, meta::Shift};
    ///
    /// let lj = Lj::<f64>::new(1.0, 3.4);
    /// let lj_shift = Shift::new(lj, 12.0);
    /// ```
    #[inline]
    pub fn new(inner: P, rc: f64) -> Self {
        let rc_sq = T::splat(rc * rc);
        let e_rc = inner.energy(rc_sq);

        Self { inner, rc_sq, e_rc }
    }

    /// Returns a reference to the inner potential.
    #[inline]
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Returns the energy shift value.
    #[inline]
    pub fn shift(&self) -> T {
        self.e_rc
    }
}

impl<P: Potential2<T>, T: Vector> Potential2<T> for Shift<P, T> {
    #[inline(always)]
    fn energy(&self, r_sq: T) -> T {
        let e = self.inner.energy(r_sq) - self.e_rc;
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
        (
            mask.select(e - self.e_rc, T::zero()),
            mask.select(s, T::zero()),
        )
    }
}
