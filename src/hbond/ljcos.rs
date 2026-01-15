//! # Lennard-Jones Cosine Potential
//!
//! A 12-6 Lennard-Jones potential with angular modulation.
//!
//! ## Formula
//!
//! ```text
//! V(r, cos_theta) = 4 * eps * [(sigma/r)^12 - (sigma/r)^6] * cos^N(theta)
//! ```
//!
//! where:
//! - `r`: Distance between interaction sites (length)
//! - `cos_theta`: Cosine of the angle (passed directly, not the angle)
//! - `eps`: Well depth (energy)
//! - `sigma`: Size parameter (length)
//! - `N`: Angular exponent (compile-time constant, typically 2 or 4)
//!
//! ## Important: Angular Exponent Parity
//!
//! The code computes `cos_theta.powi(N)` directly:
//!
//! - **Even N** (2, 4): Maximum at `cos_theta = ±1`. Works correctly for
//!   standard H-bond geometry where linear gives cos = -1.
//! - **Odd N** (1, 3): Maximum only at `cos_theta = +1`. Be aware of sign.
//!
//! ## Properties
//!
//! - Combines standard 12-6 LJ with angular dependence
//! - Minimum at r = 2^(1/6)*sigma when |cos_theta| = 1 (for even N)
//! - Angular exponent N is a compile-time constant for optimization

use crate::math::Vector;

/// Lennard-Jones with angular modulation.
///
/// ## Type Parameters
///
/// - `T`: Numeric type (f32, f64, or SIMD vector)
/// - `N`: Angular exponent (power of cos, compile-time constant)
///
/// ## Parameters
///
/// - `eps`: Well depth (energy)
/// - `sigma`: Size parameter (length)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LjCos<T, const N: u32 = 4> {
    four_eps: T,
    sigma_sq: T,
}

impl<T: Vector, const N: u32> LjCos<T, N> {
    /// Creates a new LJ-cosine potential.
    ///
    /// ## Arguments
    ///
    /// - `eps`: Well depth (energy)
    /// - `sigma`: Size parameter (length)
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::hbond::LjCos;
    ///
    /// // Typical parameters with cos^4
    /// let potential = LjCos::<f64, 4>::new(5.0, 2.5);
    ///
    /// // With cos^2 angular dependence
    /// let potential2 = LjCos::<f64, 2>::new(5.0, 2.5);
    /// ```
    #[inline]
    pub fn new(eps: f64, sigma: f64) -> Self {
        Self {
            four_eps: T::splat(4.0 * eps),
            sigma_sq: T::splat(sigma * sigma),
        }
    }

    /// Computes the potential energy.
    ///
    /// ## Arguments
    ///
    /// - `r_sq`: Squared distance (length²)
    /// - `cos_theta`: Cosine of the angle
    #[inline(always)]
    pub fn energy(&self, r_sq: T, cos_theta: T) -> T {
        // (sigma/r)^2
        let s2 = self.sigma_sq / r_sq;
        let s6 = s2 * s2 * s2;
        let s12 = s6 * s6;

        // cos^N(theta) - optimized at compile time
        let cos_n = cos_power::<T, N>(cos_theta);

        // 4*eps * (s12 - s6) * cos^N
        self.four_eps * (s12 - s6) * cos_n
    }

    /// Computes radial and angular derivatives.
    ///
    /// ## Returns
    ///
    /// - `(S, dV_dcos)` where:
    ///   - `S = -(dV/dr)/r` for force computation
    ///   - `dV_dcos` for angular force
    #[inline(always)]
    pub fn derivative(&self, r_sq: T, cos_theta: T) -> (T, T) {
        let six = T::splat(6.0);
        let twelve = T::splat(12.0);

        let s2 = self.sigma_sq / r_sq;
        let s6 = s2 * s2 * s2;
        let s12 = s6 * s6;

        let cos_n = cos_power::<T, N>(cos_theta);
        let cos_nm1 = cos_power_m1::<T, N>(cos_theta);

        // LJ radial: dV_lj/dr = 4*eps * (-12*s12/r + 6*s6/r)
        // S = -(dV/dr)/r = 4*eps * (12*s12 - 6*s6) * cos^N / r^2
        let lj_part = twelve * s12 - six * s6;
        let s = self.four_eps * lj_part * cos_n / r_sq;

        // Angular: dV/d(cos) = 4*eps * (s12 - s6) * N * cos^(N-1)
        let n_t = T::splat(N as f64);
        let lj_energy = s12 - s6;
        let dv_dcos = self.four_eps * lj_energy * n_t * cos_nm1;

        (s, dv_dcos)
    }

    /// Computes energy and both derivatives together (optimized).
    ///
    /// Shares computations for efficiency.
    #[inline(always)]
    pub fn energy_derivative(&self, r_sq: T, cos_theta: T) -> (T, T, T) {
        let six = T::splat(6.0);
        let twelve = T::splat(12.0);

        let s2 = self.sigma_sq / r_sq;
        let s6 = s2 * s2 * s2;
        let s12 = s6 * s6;

        let cos_n = cos_power::<T, N>(cos_theta);
        let cos_nm1 = cos_power_m1::<T, N>(cos_theta);

        let lj_energy_part = s12 - s6;
        let energy = self.four_eps * lj_energy_part * cos_n;

        let lj_force_part = twelve * s12 - six * s6;
        let s = self.four_eps * lj_force_part * cos_n / r_sq;

        let n_t = T::splat(N as f64);
        let dv_dcos = self.four_eps * lj_energy_part * n_t * cos_nm1;

        (energy, s, dv_dcos)
    }
}

/// Computes cos^N at compile time using const generics.
///
/// The compiler completely eliminates the match at compile time,
/// generating optimal code for each specific power.
#[inline(always)]
fn cos_power<T: Vector, const N: u32>(cos_theta: T) -> T {
    match N {
        0 => T::splat(1.0),
        1 => cos_theta,
        2 => cos_theta * cos_theta,
        3 => cos_theta * cos_theta * cos_theta,
        4 => {
            let c2 = cos_theta * cos_theta;
            c2 * c2
        }
        5 => {
            let c2 = cos_theta * cos_theta;
            c2 * c2 * cos_theta
        }
        6 => {
            let c2 = cos_theta * cos_theta;
            c2 * c2 * c2
        }
        _ => {
            // Binary exponentiation for large N
            let mut result = T::splat(1.0);
            let mut base = cos_theta;
            let mut exp = N;
            while exp > 0 {
                if exp & 1 == 1 {
                    result = result * base;
                }
                base = base * base;
                exp >>= 1;
            }
            result
        }
    }
}

/// Computes cos^(N-1) at compile time.
///
/// Separate function to avoid `{ N - 1 }` in generic position.
#[inline(always)]
fn cos_power_m1<T: Vector, const N: u32>(cos_theta: T) -> T {
    match N {
        0 | 1 => T::splat(1.0),
        2 => cos_theta,
        3 => cos_theta * cos_theta,
        4 => cos_theta * cos_theta * cos_theta,
        5 => {
            let c2 = cos_theta * cos_theta;
            c2 * c2
        }
        6 => {
            let c2 = cos_theta * cos_theta;
            c2 * c2 * cos_theta
        }
        7 => {
            let c2 = cos_theta * cos_theta;
            c2 * c2 * c2
        }
        _ => {
            let mut result = T::splat(1.0);
            let mut base = cos_theta;
            let mut exp = N - 1;
            while exp > 0 {
                if exp & 1 == 1 {
                    result = result * base;
                }
                base = base * base;
                exp >>= 1;
            }
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ljcos_at_minimum() {
        let eps = 5.0;
        let sigma = 2.5;
        let ljcos: LjCos<f64, 4> = LjCos::new(eps, sigma);

        let r_min = 2.0_f64.powf(1.0 / 6.0) * sigma;
        let e = ljcos.energy(r_min * r_min, 1.0);

        assert_relative_eq!(e, -eps, epsilon = 1e-10);
    }

    #[test]
    fn test_ljcos_angular_at_90() {
        let ljcos: LjCos<f64, 4> = LjCos::new(5.0, 2.5);

        let e = ljcos.energy(3.0 * 3.0, 0.0);
        assert_relative_eq!(e, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ljcos_cos2_vs_cos4() {
        let eps = 5.0;
        let sigma = 2.5;
        let cos2: LjCos<f64, 2> = LjCos::new(eps, sigma);
        let cos4: LjCos<f64, 4> = LjCos::new(eps, sigma);

        let r_sq = 3.0 * 3.0;
        let cos_theta = 0.7;

        let e2 = cos2.energy(r_sq, cos_theta);
        let e4 = cos4.energy(r_sq, cos_theta);

        assert!(e4.abs() < e2.abs());

        let e_lj = cos2.energy(r_sq, 1.0);
        let ratio2 = e2 / e_lj;
        let ratio4 = e4 / e_lj;

        assert_relative_eq!(ratio2, cos_theta * cos_theta, epsilon = 1e-10);
        assert_relative_eq!(ratio4, cos_theta.powi(4), epsilon = 1e-10);
    }

    #[test]
    fn test_ljcos_numerical_radial_derivative() {
        let ljcos: LjCos<f64, 4> = LjCos::new(5.0, 2.5);
        let r = 3.0;
        let cos_theta = 0.8;

        let h = 1e-7;
        let e_plus = ljcos.energy((r + h) * (r + h), cos_theta);
        let e_minus = ljcos.energy((r - h) * (r - h), cos_theta);
        let dv_dr_numerical = (e_plus - e_minus) / (2.0 * h);

        let (s, _) = ljcos.derivative(r * r, cos_theta);
        let dv_dr_analytical = -s * r;

        assert_relative_eq!(dv_dr_analytical, dv_dr_numerical, epsilon = 1e-6);
    }

    #[test]
    fn test_ljcos_numerical_angular_derivative() {
        let ljcos: LjCos<f64, 4> = LjCos::new(5.0, 2.5);
        let r_sq = 3.0 * 3.0;
        let cos_theta = 0.75;

        let h = 1e-7;
        let e_plus = ljcos.energy(r_sq, cos_theta + h);
        let e_minus = ljcos.energy(r_sq, cos_theta - h);
        let dv_dcos_numerical = (e_plus - e_minus) / (2.0 * h);

        let (_, dv_dcos_analytical) = ljcos.derivative(r_sq, cos_theta);

        assert_relative_eq!(dv_dcos_analytical, dv_dcos_numerical, epsilon = 1e-6);
    }

    #[test]
    fn test_cos_power() {
        let c = 0.7_f64;

        assert_relative_eq!(cos_power::<f64, 0>(c), 1.0, epsilon = 1e-10);
        assert_relative_eq!(cos_power::<f64, 1>(c), c, epsilon = 1e-10);
        assert_relative_eq!(cos_power::<f64, 2>(c), c * c, epsilon = 1e-10);
        assert_relative_eq!(cos_power::<f64, 4>(c), c.powi(4), epsilon = 1e-10);
        assert_relative_eq!(cos_power::<f64, 6>(c), c.powi(6), epsilon = 1e-10);
    }

    #[test]
    fn test_ljcos_energy_derivative_consistency() {
        let ljcos: LjCos<f64, 3> = LjCos::new(4.0, 2.2);
        let r_sq = 2.8 * 2.8;
        let cos_theta = 0.65;

        let e1 = ljcos.energy(r_sq, cos_theta);
        let (s1, dc1) = ljcos.derivative(r_sq, cos_theta);
        let (e2, s2, dc2) = ljcos.energy_derivative(r_sq, cos_theta);

        assert_relative_eq!(e1, e2, epsilon = 1e-10);
        assert_relative_eq!(s1, s2, epsilon = 1e-10);
        assert_relative_eq!(dc1, dc2, epsilon = 1e-10);
    }

    #[test]
    fn test_ljcos_various_powers() {
        let eps = 5.0;
        let sigma = 2.5;
        let r_sq = 3.0 * 3.0;
        let cos_theta = 0.8;

        let ljcos2: LjCos<f64, 2> = LjCos::new(eps, sigma);
        let ljcos4: LjCos<f64, 4> = LjCos::new(eps, sigma);
        let ljcos6: LjCos<f64, 6> = LjCos::new(eps, sigma);

        let e2 = ljcos2.energy(r_sq, cos_theta);
        let e4 = ljcos4.energy(r_sq, cos_theta);
        let e6 = ljcos6.energy(r_sq, cos_theta);

        let e_lin = ljcos2.energy(r_sq, 1.0);

        assert_relative_eq!(e2 / e_lin, cos_theta.powi(2), epsilon = 1e-10);
        assert_relative_eq!(e4 / e_lin, cos_theta.powi(4), epsilon = 1e-10);
        assert_relative_eq!(e6 / e_lin, cos_theta.powi(6), epsilon = 1e-10);
    }
}
