//! # Generalized Mie Potential (n-m)
//!
//! The Mie potential generalizes Lennard-Jones with arbitrary integer exponents.
//!
//! ## Formula
//!
//! ```text
//! V(r) = C * epsilon * [(sigma/r)^n - (sigma/r)^m]
//!
//! where:
//!     C = (n / (n - m)) * (n / m)^(m / (n - m))
//! ```
//!
//! The prefactor C ensures that:
//! - The minimum energy is exactly -epsilon
//! - r_min = (n/m)^(1/(n-m)) * sigma
//!
//! ## Special Cases
//!
//! | N  | M | Name |
//! |----|---|------|
//! | 12 | 6 | Lennard-Jones |
//! | 9  | 6 | Soft-core LJ |
//! | 14 | 7 | Harder repulsion |
//!
//! ## Implementation Notes
//!
//! - Uses const generics for compile-time optimization of exponents
//! - Exponent computations are fully unrolled by the compiler
//! - Zero runtime overhead for exponent handling

use crate::base::Potential2;
use crate::math::Vector;

/// Generalized Mie (n-m) potential.
///
/// ## Type Parameters
///
/// - `T`: Numeric type (f32, f64, or SIMD vector)
/// - `N`: Repulsive exponent (must be > M)
/// - `M`: Attractive exponent (must be > 0)
///
/// ## Parameters
///
/// - `epsilon`: Well depth (energy)
/// - `sigma`: Characteristic length (length)
///
/// ## Precomputed Values
///
/// Stores `cn = C * epsilon * sigma^n` and `cm = C * epsilon * sigma^m`
/// for efficient evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mie<T, const N: u32, const M: u32> {
    /// Coefficient for r^-n term: C * epsilon * sigma^n
    cn: T,
    /// Coefficient for r^-m term: C * epsilon * sigma^m
    cm: T,
}

impl<T: Vector, const N: u32, const M: u32> Mie<T, N, M> {
    /// Creates a new Mie potential.
    ///
    /// ## Arguments
    ///
    /// - `epsilon`: Well depth (energy units)
    /// - `sigma`: Characteristic length (length units)
    ///
    /// ## Panics
    ///
    /// Panics if `N <= M` or `M == 0`.
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::pair::Mie;
    ///
    /// // Standard LJ 12-6
    /// let lj = Mie::<f64, 12, 6>::new(1.0, 1.0);
    ///
    /// // Softer 9-6 potential
    /// let soft = Mie::<f64, 9, 6>::new(1.0, 1.0);
    /// ```
    #[inline]
    pub fn new(epsilon: f64, sigma: f64) -> Self {
        assert!(N > M, "Mie potential requires N > M, got N={}, M={}", N, M);
        assert!(M > 0, "Mie potential requires M > 0, got M={}", M);

        let n = N as f64;
        let m = M as f64;

        // Prefactor: C = (n/(n-m)) * (n/m)^(m/(n-m))
        let n_minus_m = n - m;
        let prefactor = (n / n_minus_m) * (n / m).powf(m / n_minus_m);
        let c_eps = prefactor * epsilon;

        // Precompute sigma^n and sigma^m using integer powers
        let sigma_n = int_pow(sigma, N);
        let sigma_m = int_pow(sigma, M);

        Self {
            cn: T::splat(c_eps * sigma_n),
            cm: T::splat(c_eps * sigma_m),
        }
    }

    /// Creates a new Mie potential from precomputed coefficients.
    ///
    /// ## Arguments
    ///
    /// - `cn`: Coefficient for r^-n term (C * epsilon * sigma^n)
    /// - `cm`: Coefficient for r^-m term (C * epsilon * sigma^m)
    ///
    /// This is useful when mixing rules produce cn/cm directly.
    #[inline]
    pub fn from_coefficients(cn: f64, cm: f64) -> Self {
        Self {
            cn: T::splat(cn),
            cm: T::splat(cm),
        }
    }

    /// Returns the cn coefficient.
    #[inline]
    pub fn cn(&self) -> T {
        self.cn
    }

    /// Returns the cm coefficient.
    #[inline]
    pub fn cm(&self) -> T {
        self.cm
    }

    /// Returns the repulsive exponent N.
    #[inline]
    pub const fn n(&self) -> u32 {
        N
    }

    /// Returns the attractive exponent M.
    #[inline]
    pub const fn m(&self) -> u32 {
        M
    }
}

impl<T: Vector, const N: u32, const M: u32> Potential2<T> for Mie<T, N, M> {
    /// Computes the potential energy.
    ///
    /// ```text
    /// V(r) = cn/r^n - cm/r^m
    /// ```
    #[inline(always)]
    fn energy(&self, r_sq: T) -> T {
        let r_inv = r_sq.rsqrt();

        // Compute r^-n and r^-m using optimized integer powers
        let r_neg_n = int_pow_vec::<T, N>(r_inv);
        let r_neg_m = int_pow_vec::<T, M>(r_inv);

        self.cn * r_neg_n - self.cm * r_neg_m
    }

    /// Computes the force factor.
    ///
    /// ```text
    /// dV/dr = -n*cn/r^(n+1) + m*cm/r^(m+1)
    /// S = -(dV/dr)/r = n*cn/r^(n+2) - m*cm/r^(m+2)
    /// ```
    #[inline(always)]
    fn force_factor(&self, r_sq: T) -> T {
        let r_inv = r_sq.rsqrt();
        let r_sq_inv = r_sq.recip();

        let r_neg_n = int_pow_vec::<T, N>(r_inv);
        let r_neg_m = int_pow_vec::<T, M>(r_inv);

        let n = T::splat(N as f64);
        let m = T::splat(M as f64);

        (n * self.cn * r_neg_n - m * self.cm * r_neg_m) * r_sq_inv
    }

    /// Computes energy and force factor together (optimized).
    ///
    /// Shares the computation of r^-n and r^-m.
    #[inline(always)]
    fn energy_force(&self, r_sq: T) -> (T, T) {
        let r_inv = r_sq.rsqrt();
        let r_sq_inv = r_sq.recip();

        let r_neg_n = int_pow_vec::<T, N>(r_inv);
        let r_neg_m = int_pow_vec::<T, M>(r_inv);

        let cn_rn = self.cn * r_neg_n;
        let cm_rm = self.cm * r_neg_m;

        let energy = cn_rn - cm_rm;

        let n = T::splat(N as f64);
        let m = T::splat(M as f64);
        let force = (n * cn_rn - m * cm_rm) * r_sq_inv;

        (energy, force)
    }
}

/// Computes x^n using binary exponentiation for f64.
///
/// Used for precomputation of sigma^n in constructor.
#[inline]
fn int_pow(x: f64, n: u32) -> f64 {
    match n {
        0 => 1.0,
        1 => x,
        2 => x * x,
        3 => x * x * x,
        4 => {
            let x2 = x * x;
            x2 * x2
        }
        5 => {
            let x2 = x * x;
            x2 * x2 * x
        }
        6 => {
            let x2 = x * x;
            x2 * x2 * x2
        }
        7 => {
            let x2 = x * x;
            let x4 = x2 * x2;
            x4 * x2 * x
        }
        8 => {
            let x2 = x * x;
            let x4 = x2 * x2;
            x4 * x4
        }
        9 => {
            let x2 = x * x;
            let x4 = x2 * x2;
            let x8 = x4 * x4;
            x8 * x
        }
        10 => {
            let x2 = x * x;
            let x4 = x2 * x2;
            let x8 = x4 * x4;
            x8 * x2
        }
        11 => {
            let x2 = x * x;
            let x4 = x2 * x2;
            let x8 = x4 * x4;
            x8 * x2 * x
        }
        12 => {
            let x2 = x * x;
            let x4 = x2 * x2;
            x4 * x4 * x4
        }
        13 => {
            let x2 = x * x;
            let x4 = x2 * x2;
            x4 * x4 * x4 * x
        }
        14 => {
            let x2 = x * x;
            let x4 = x2 * x2;
            let x8 = x4 * x4;
            x8 * x4 * x2
        }
        _ => {
            // Binary exponentiation for large n
            let mut result = 1.0;
            let mut base = x;
            let mut exp = n;
            while exp > 0 {
                if exp & 1 == 1 {
                    result *= base;
                }
                base *= base;
                exp >>= 1;
            }
            result
        }
    }
}

/// Computes x^N for Vector types at compile time using const generics.
///
/// The compiler completely eliminates the match at compile time,
/// generating optimal code for each specific power.
#[inline(always)]
fn int_pow_vec<T: Vector, const N: u32>(x: T) -> T {
    match N {
        0 => T::splat(1.0),
        1 => x,
        2 => x * x,
        3 => x * x * x,
        4 => {
            let x2 = x * x;
            x2 * x2
        }
        5 => {
            let x2 = x * x;
            x2 * x2 * x
        }
        6 => {
            let x2 = x * x;
            x2 * x2 * x2
        }
        7 => {
            let x2 = x * x;
            let x4 = x2 * x2;
            x4 * x2 * x
        }
        8 => {
            let x2 = x * x;
            let x4 = x2 * x2;
            x4 * x4
        }
        9 => {
            let x2 = x * x;
            let x4 = x2 * x2;
            let x8 = x4 * x4;
            x8 * x
        }
        10 => {
            let x2 = x * x;
            let x4 = x2 * x2;
            let x8 = x4 * x4;
            x8 * x2
        }
        11 => {
            let x2 = x * x;
            let x4 = x2 * x2;
            let x8 = x4 * x4;
            x8 * x2 * x
        }
        12 => {
            let x2 = x * x;
            let x4 = x2 * x2;
            x4 * x4 * x4
        }
        13 => {
            let x2 = x * x;
            let x4 = x2 * x2;
            x4 * x4 * x4 * x
        }
        14 => {
            let x2 = x * x;
            let x4 = x2 * x2;
            let x8 = x4 * x4;
            x8 * x4 * x2
        }
        _ => {
            // Binary exponentiation for large N
            let mut result = T::splat(1.0);
            let mut base = x;
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mie_reduces_to_lj() {
        let mie: Mie<f64, 12, 6> = Mie::new(1.0, 1.0);
        let lj = crate::pair::Lj::<f64>::new(1.0, 1.0);

        let r_sq = 1.5 * 1.5;
        let e_mie = mie.energy(r_sq);
        let e_lj = lj.energy(r_sq);

        assert_relative_eq!(e_mie, e_lj, epsilon = 1e-10);
    }

    #[test]
    fn test_mie_at_minimum() {
        let epsilon = 1.5;
        let sigma = 2.0;
        let mie: Mie<f64, 10, 5> = Mie::new(epsilon, sigma);

        let r_min = 2.0_f64.powf(1.0 / 5.0) * sigma;
        let r_sq = r_min * r_min;
        let energy = mie.energy(r_sq);

        assert_relative_eq!(energy, -epsilon, epsilon = 1e-10);
    }

    #[test]
    fn test_mie_force_at_minimum() {
        let mie: Mie<f64, 12, 6> = Mie::new(1.0, 1.0);

        let r_min = 2.0_f64.powf(1.0 / 6.0);
        let r_sq = r_min * r_min;
        let force = mie.force_factor(r_sq);

        assert_relative_eq!(force, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mie_energy_force_consistency() {
        let mie: Mie<f64, 9, 6> = Mie::new(0.5, 2.0);
        let r_sq = 6.25;

        let (e1, f1) = mie.energy_force(r_sq);
        let e2 = mie.energy(r_sq);
        let f2 = mie.force_factor(r_sq);

        assert_relative_eq!(e1, e2, epsilon = 1e-12);
        assert_relative_eq!(f1, f2, epsilon = 1e-12);
    }

    #[test]
    fn test_mie_numerical_derivative() {
        let mie: Mie<f64, 9, 6> = Mie::new(1.0, 1.0);
        let r = 1.2;
        let r_sq = r * r;

        let h = 1e-6;
        let v_plus = mie.energy((r + h) * (r + h));
        let v_minus = mie.energy((r - h) * (r - h));
        let dv_dr_numerical = (v_plus - v_minus) / (2.0 * h);

        let s_numerical = -dv_dr_numerical / r;
        let s_analytical = mie.force_factor(r_sq);

        assert_relative_eq!(s_analytical, s_numerical, epsilon = 1e-6);
    }
}
