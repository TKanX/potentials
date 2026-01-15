//! # Switching Function Wrapper
//!
//! Smoothly turns off potential between switch distance and cutoff.
//!
//! ## Formula
//!
//! For rs <= r <= rc:
//!
//! ```text
//! S(r) = [(rc^2 - r^2)^2 * (rc^2 + 2*r^2 - 3*rs^2)] / [(rc^2 - rs^2)^3]
//!
//! V_switch(r) = V(r) * S(r)
//! ```
//!
//! The switching function S(r) satisfies:
//! - S(rs) = 1
//! - S(rc) = 0
//! - S'(rs) = 0
//! - S'(rc) = 0
//!
//! This ensures both energy and force go smoothly to zero.
//!
//! ## Notes
//!
//! - Both energy and force are continuous
//! - Common in CHARMM and LAMMPS
//! - More computationally expensive than simple shift

use crate::base::Potential2;
use crate::math::{Mask, Vector};

/// Switching function wrapper.
///
/// ## Type Parameters
///
/// - `P`: The underlying potential type
/// - `T`: The vector type
#[derive(Clone, Copy, Debug)]
pub struct Switch<P, T> {
    inner: P,
    rs_sq: T,
    rc_sq: T,
    inv_denom: T, // 1 / (rc^2 - rs^2)^3
    three_rs_sq: T,
}

impl<P, T: Vector> Switch<P, T> {
    /// Creates a new switching function wrapper.
    ///
    /// ## Arguments
    ///
    /// - `inner`: The potential to wrap
    /// - `rs`: Switch-on distance (where switching starts) (length)
    /// - `rc`: Cutoff distance (where potential is zero) (length)
    ///
    /// ## Panics
    ///
    /// In debug mode, panics if rs >= rc.
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::{pair::Lj, meta::Switch};
    ///
    /// let lj = Lj::<f64>::new(1.0, 3.4);
    /// let lj_switch: Switch<_, f64> = Switch::new(lj, 9.0, 12.0);
    /// ```
    #[inline]
    pub fn new(inner: P, rs: f64, rc: f64) -> Self {
        debug_assert!(rs < rc, "Switch distance must be less than cutoff");

        let rs_sq = rs * rs;
        let rc_sq = rc * rc;
        let diff = rc_sq - rs_sq;
        let denom = diff * diff * diff;

        Self {
            inner,
            rs_sq: T::splat(rs_sq),
            rc_sq: T::splat(rc_sq),
            inv_denom: T::splat(1.0 / denom),
            three_rs_sq: T::splat(3.0 * rs_sq),
        }
    }

    /// Computes the switching function value.
    #[inline(always)]
    fn switch_value(&self, r_sq: T) -> T {
        let two = T::splat(2.0);

        // (rc^2 - r^2)^2
        let rc_minus_r = self.rc_sq - r_sq;
        let term1 = rc_minus_r * rc_minus_r;

        // (rc^2 + 2*r^2 - 3*rs^2)
        let term2 = self.rc_sq + two * r_sq - self.three_rs_sq;

        term1 * term2 * self.inv_denom
    }

    /// Computes the derivative of switching function dS/d(r^2).
    #[inline(always)]
    fn switch_derivative(&self, r_sq: T) -> T {
        let six = T::splat(6.0);

        // dS/d(r^2) = 6 * (rc^2 - r^2) * (rs^2 - r^2) / denom
        let term = six * (self.rc_sq - r_sq) * (self.rs_sq - r_sq);
        term * self.inv_denom
    }
}

impl<P: Potential2<T>, T: Vector> Potential2<T> for Switch<P, T> {
    #[inline(always)]
    fn energy(&self, r_sq: T) -> T {
        let e = self.inner.energy(r_sq);

        // r < rs: full potential
        let inside_rs = r_sq.lt(self.rs_sq);

        // rs <= r < rc: switched potential
        let inside_rc = r_sq.lt(self.rc_sq);
        let in_switch = inside_rc & !inside_rs;

        let s = self.switch_value(r_sq);

        // Combine: inside_rs -> e, in_switch -> e*s, outside -> 0
        inside_rs.select(e, in_switch.select(e * s, T::zero()))
    }

    #[inline(always)]
    fn force_factor(&self, r_sq: T) -> T {
        let two = T::splat(2.0);

        let e = self.inner.energy(r_sq);
        let f = self.inner.force_factor(r_sq);

        let inside_rs = r_sq.lt(self.rs_sq);
        let inside_rc = r_sq.lt(self.rc_sq);
        let in_switch = inside_rc & !inside_rs;

        let s_func = self.switch_value(r_sq);
        let ds_dr2 = self.switch_derivative(r_sq);

        let f_switched = f * s_func - two * e * ds_dr2;

        inside_rs.select(f, in_switch.select(f_switched, T::zero()))
    }

    #[inline(always)]
    fn energy_force(&self, r_sq: T) -> (T, T) {
        let two = T::splat(2.0);

        let (e, f) = self.inner.energy_force(r_sq);

        let inside_rs = r_sq.lt(self.rs_sq);
        let inside_rc = r_sq.lt(self.rc_sq);
        let in_switch = inside_rc & !inside_rs;

        let s_func = self.switch_value(r_sq);
        let ds_dr2 = self.switch_derivative(r_sq);

        let e_switched = e * s_func;
        let f_switched = f * s_func - two * e * ds_dr2;

        (
            inside_rs.select(e, in_switch.select(e_switched, T::zero())),
            inside_rs.select(f, in_switch.select(f_switched, T::zero())),
        )
    }
}
