//! # Soft Sphere Potential
//!
//! A purely repulsive power-law potential commonly used for
//! soft matter simulations and as a WCA-style short-range repulsion.
//!
//! ## Formula
//!
//! ```text
//! V(r) = epsilon * (sigma / r)^n
//! ```
//!
//! where:
//! - `epsilon`: Energy scale (energy units)
//! - `sigma`: Length scale (length units)
//! - `N`: Repulsion exponent (compile-time constant)
//!
//! ## Force Factor
//!
//! ```text
//! S = -(dV/dr) / r = n * epsilon * sigma^n / r^(n+2)
//! ```
//!
//! ## Implementation Notes
//!
//! - Uses const generics for compile-time optimization of exponent
//! - Exponent computation is fully unrolled by the compiler
//! - Stores `coeff = epsilon * sigma^n` for efficient evaluation

use crate::base::Potential2;
use crate::math::Vector;

/// Soft sphere (power-law repulsion) potential.
///
/// ## Type Parameters
///
/// - `T`: Numeric type (f32, f64, or SIMD vector)
/// - `N`: Repulsion exponent (must be > 0)
///
/// ## Parameters
///
/// - `epsilon`: Energy scale (energy units)
/// - `sigma`: Length scale (length units)
///
/// ## Precomputed Values
///
/// - `coeff`: Stores `epsilon * sigma^n` for efficient evaluation
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Soft<T, const N: u32> {
    /// Precomputed coefficient: epsilon * sigma^n
    coeff: T,
}

impl<T: Vector, const N: u32> Soft<T, N> {
    /// Creates a new soft sphere potential.
    ///
    /// ## Arguments
    ///
    /// - `epsilon`: Energy scale (energy units)
    /// - `sigma`: Length scale (length units)
    ///
    /// ## Panics
    ///
    /// Panics if `N == 0` (compile-time enforced where possible).
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::pair::Soft;
    ///
    /// // Hard-sphere-like repulsion (n=12)
    /// let soft = Soft::<f64, 12>::new(1.0, 1.0);
    ///
    /// // Softer repulsion (n=6)
    /// let soft6 = Soft::<f64, 6>::new(1.0, 1.0);
    /// ```
    #[inline]
    pub fn new(epsilon: f64, sigma: f64) -> Self {
        assert!(N > 0, "Soft sphere exponent must be positive, got N={}", N);

        let sigma_n = int_pow(sigma, N);
        Self {
            coeff: T::splat(epsilon * sigma_n),
        }
    }

    /// Creates a soft sphere potential from a precomputed coefficient.
    ///
    /// ## Arguments
    ///
    /// - `coeff`: Precomputed coefficient (epsilon * sigma^n)
    ///
    /// This is useful when mixing rules produce the coefficient directly.
    #[inline]
    pub fn from_coefficient(coeff: f64) -> Self {
        Self {
            coeff: T::splat(coeff),
        }
    }

    /// Returns the coefficient (epsilon * sigma^n).
    #[inline]
    pub fn coeff(&self) -> T {
        self.coeff
    }

    /// Returns the exponent N.
    #[inline]
    pub const fn exponent(&self) -> u32 {
        N
    }
}

impl<T: Vector, const N: u32> Potential2<T> for Soft<T, N> {
    /// Computes the potential energy.
    ///
    /// ```text
    /// V(r) = coeff / r^n
    /// ```
    #[inline(always)]
    fn energy(&self, r_sq: T) -> T {
        let r_inv = r_sq.rsqrt();
        let r_neg_n = int_pow_vec::<T, N>(r_inv);
        self.coeff * r_neg_n
    }

    /// Computes the force factor.
    ///
    /// ```text
    /// dV/dr = -n * coeff / r^(n+1)
    /// S = -(dV/dr)/r = n * coeff / r^(n+2)
    /// ```
    #[inline(always)]
    fn force_factor(&self, r_sq: T) -> T {
        let r_inv = r_sq.rsqrt();
        let r_neg_n = int_pow_vec::<T, N>(r_inv);
        let r_sq_inv = r_sq.recip();
        let n = T::splat(N as f64);

        n * self.coeff * r_neg_n * r_sq_inv
    }

    /// Computes energy and force factor together (optimized).
    ///
    /// Shares the computation of r^-n.
    #[inline(always)]
    fn energy_force(&self, r_sq: T) -> (T, T) {
        let r_inv = r_sq.rsqrt();
        let r_neg_n = int_pow_vec::<T, N>(r_inv);
        let r_sq_inv = r_sq.recip();

        let coeff_rn = self.coeff * r_neg_n;
        let energy = coeff_rn;

        let n = T::splat(N as f64);
        let force = n * coeff_rn * r_sq_inv;

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
