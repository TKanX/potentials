//! # Urey-Bradley Angle Potential
//!
//! A combined potential with both angle bending and 1-3 distance terms.
//!
//! ## Formula
//!
//! ```text
//! V(theta, r_ik) = k_theta * (theta - theta0)^2 + k_ub * (r_ik - r_ub)^2
//! ```
//!
//! where:
//! - `k_theta`: Angular force constant (energy/radian^2 units)
//! - `theta0`: Equilibrium angle (radians)
//! - `k_ub`: Urey-Bradley force constant (energy/length^2 units)
//! - `r_ub`: Equilibrium 1-3 distance (length units)
//!
//! ## Implementation Notes
//!
//! - Uses cosine form for efficiency: V = k_cos * (cos - cos0)^2 + k_ub * (r_ik - r_ub)^2
//! - Used in CHARMM force field
//! - r_ik computed via law of cosines: r_ik^2 = r_ij^2 + r_jk^2 - 2*r_ij*r_jk*cos(theta)

use crate::base::Potential3;
use crate::math::Vector;

/// Urey-Bradley angle potential.
///
/// ## Parameters
///
/// - `k_cos`: Angular force constant, cosine form (energy units)
/// - `cos0`: Equilibrium angle cosine
/// - `k_ub`: Urey-Bradley force constant (energy/length^2 units)
/// - `r_ub`: Equilibrium 1-3 distance (length units)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Urey<T> {
    /// Angular force constant (cosine form)
    k_cos: T,
    /// Equilibrium angle cosine
    cos0: T,
    /// Urey-Bradley force constant
    k_ub: T,
    /// Equilibrium 1-3 distance
    r_ub: T,
}

impl<T: Vector> Urey<T> {
    /// Creates a new Urey-Bradley potential (cosine form).
    ///
    /// ## Arguments
    ///
    /// - `k_cos`: Angular force constant (cosine form)
    /// - `cos0`: Equilibrium angle cosine
    /// - `k_ub`: Urey-Bradley force constant
    /// - `r_ub`: Equilibrium 1-3 distance
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::angle::Urey;
    ///
    /// let cos0 = (109.5_f64 * std::f64::consts::PI / 180.0).cos();
    /// let angle = Urey::<f64>::new(100.0, cos0, 50.0, 2.4);
    /// ```
    #[inline]
    pub fn new(k_cos: f64, cos0: f64, k_ub: f64, r_ub: f64) -> Self {
        Self {
            k_cos: T::splat(k_cos),
            cos0: T::splat(cos0),
            k_ub: T::splat(k_ub),
            r_ub: T::splat(r_ub),
        }
    }

    /// Creates from angle in degrees.
    #[inline]
    pub fn from_degrees(k_cos: f64, theta0_deg: f64, k_ub: f64, r_ub: f64) -> Self {
        let theta0_rad = theta0_deg * core::f64::consts::PI / 180.0;
        Self::new(k_cos, theta0_rad.cos(), k_ub, r_ub)
    }

    /// Returns the angular force constant.
    #[inline]
    pub fn k_cos(&self) -> T {
        self.k_cos
    }

    /// Returns the Urey-Bradley force constant.
    #[inline]
    pub fn k_ub(&self) -> T {
        self.k_ub
    }
}

impl<T: Vector> Potential3<T> for Urey<T> {
    /// Computes the potential energy.
    ///
    /// ```text
    /// V = k_cos * (cos - cos0)^2 + k_ub * (r_ik - r_ub)^2
    /// ```
    ///
    /// where r_ik is computed from law of cosines.
    #[inline(always)]
    fn energy(&self, r_ij_sq: T, r_jk_sq: T, cos_theta: T) -> T {
        // Angular term (cosine form)
        let delta_cos = cos_theta - self.cos0;
        let e_angle = self.k_cos * delta_cos * delta_cos;

        // 1-3 distance from law of cosines: r_ik^2 = r_ij^2 + r_jk^2 - 2*r_ij*r_jk*cos
        let r_ij = r_ij_sq.sqrt();
        let r_jk = r_jk_sq.sqrt();
        let two = T::splat(2.0);
        let r_ik_sq = r_ij_sq + r_jk_sq - two * r_ij * r_jk * cos_theta;
        let r_ik = r_ik_sq.sqrt();

        // Urey-Bradley term
        let delta_r = r_ik - self.r_ub;
        let e_ub = self.k_ub * delta_r * delta_r;

        e_angle + e_ub
    }

    /// Computes dV/d(cos_theta).
    ///
    /// Both the angular and UB terms contribute.
    #[inline(always)]
    fn derivative(&self, r_ij_sq: T, r_jk_sq: T, cos_theta: T) -> T {
        // Angular term derivative
        let delta_cos = cos_theta - self.cos0;
        let two = T::splat(2.0);
        let d_angle = two * self.k_cos * delta_cos;

        // UB term: need d(r_ik)/d(cos_theta)
        let r_ij = r_ij_sq.sqrt();
        let r_jk = r_jk_sq.sqrt();
        let r_ik_sq = r_ij_sq + r_jk_sq - two * r_ij * r_jk * cos_theta;
        let r_ik = r_ik_sq.sqrt();

        // d(r_ik)/d(cos) = -r_ij * r_jk / r_ik
        let dr_ik_dcos = T::zero() - r_ij * r_jk / r_ik;

        let delta_r = r_ik - self.r_ub;
        let d_ub = two * self.k_ub * delta_r * dr_ik_dcos;

        d_angle + d_ub
    }

    /// Computes energy and derivative together (optimized).
    ///
    /// Shares the computation of `r_ij`, `r_jk`, `r_ik`, and `delta` terms.
    #[inline(always)]
    fn energy_derivative(&self, r_ij_sq: T, r_jk_sq: T, cos_theta: T) -> (T, T) {
        let two = T::splat(2.0);

        // Common computations
        let r_ij = r_ij_sq.sqrt();
        let r_jk = r_jk_sq.sqrt();
        let r_ij_r_jk = r_ij * r_jk;
        let r_ik_sq = r_ij_sq + r_jk_sq - two * r_ij_r_jk * cos_theta;
        let r_ik = r_ik_sq.sqrt();

        // Angular term
        let delta_cos = cos_theta - self.cos0;
        let e_angle = self.k_cos * delta_cos * delta_cos;
        let d_angle = two * self.k_cos * delta_cos;

        // UB term
        let delta_r = r_ik - self.r_ub;
        let e_ub = self.k_ub * delta_r * delta_r;

        let dr_ik_dcos = T::zero() - r_ij_r_jk / r_ik;
        let d_ub = two * self.k_ub * delta_r * dr_ik_dcos;

        (e_angle + e_ub, d_angle + d_ub)
    }
}
