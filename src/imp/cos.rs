//! # Periodic Cosine Improper Torsion
//!
//! AMBER/CHARMM-style periodic potential for improper dihedrals.
//!
//! ## Formula
//!
//! ```text
//! V(xi) = k * (1 + cos(n*xi - d))
//! ```
//!
//! where:
//! - `xi`: Current improper angle (radians)
//! - `k`: Force constant (energy units)
//! - `n`: Periodicity (multiplicity), typically 2
//! - `d`: Phase shift (radians)
//!
//! ## Derivative
//!
//! ```text
//! dV/d(xi) = -k * n * sin(n*xi - d)
//! ```
//!
//! ## Implementation Notes
//!
//! - Planar sp2: n=2, d=π (minima at 0, ±π)
//! - Trigonal: n=3, d=0 (three equivalent minima)
//! - Used by AMBER for aromatic hydrogens

use crate::base::Potential4;
use crate::math::Vector;

/// Periodic cosine improper torsion potential.
///
/// ## Parameters
///
/// - `k`: Force constant (energy units)
/// - `n`: Periodicity (integer)
/// - `d`: Phase shift (radians)
///
/// ## Precomputed Values
///
/// - `cos_d`, `sin_d`: Precomputed for angle-subtraction formula
/// - `neg_n`: Precomputed `-n` for derivative efficiency
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cos<T> {
    k: T,
    n: i32,
    cos_d: T,
    sin_d: T,
    neg_n: T,
}

impl<T: Vector> Cos<T> {
    /// Creates a new periodic improper potential.
    ///
    /// ## Arguments
    ///
    /// - `k`: Force constant (energy units)
    /// - `n`: Periodicity (1, 2, 3, ...)
    /// - `d`: Phase shift (radians)
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::imp::Cos;
    /// use core::f64::consts::PI;
    ///
    /// // Planar improper (n=2, d=pi for minimum at 0)
    /// let planar = Cos::<f64>::planar(10.0);
    /// ```
    #[inline]
    pub fn new(k: f64, n: i32, d: f64) -> Self {
        Self {
            k: T::splat(k),
            n,
            cos_d: T::splat(d.cos()),
            sin_d: T::splat(d.sin()),
            neg_n: T::splat(-(n as f64)),
        }
    }

    /// Creates for planar geometry (n=2, d=π).
    ///
    /// Minimum at xi = 0 and xi = ±π
    #[inline]
    pub fn planar(k: f64) -> Self {
        Self::new(k, 2, core::f64::consts::PI)
    }

    /// Creates for trigonal geometry (n=3, d=0).
    ///
    /// Three equivalent minima at 0, ±120°
    #[inline]
    pub fn trigonal(k: f64) -> Self {
        Self::new(k, 3, 0.0)
    }
}

impl<T: Vector> Potential4<T> for Cos<T> {
    /// Computes the potential energy using Chebyshev recursion for cos(n*xi).
    ///
    /// ```text
    /// V = k * (1 + cos(n*xi - d))
    ///   = k * (1 + cos(n*xi)*cos(d) + sin(n*xi)*sin(d))
    /// ```
    #[inline(always)]
    fn energy(&self, cos_xi: T, sin_xi: T) -> T {
        let one = T::splat(1.0);

        // Compute cos(n*xi) and sin(n*xi) using Chebyshev recursion
        let (cos_n, sin_n) = chebyshev_cos_sin(self.n, cos_xi, sin_xi);

        // cos(n*xi - d) = cos(n*xi)*cos(d) + sin(n*xi)*sin(d)
        let cos_term = cos_n * self.cos_d + sin_n * self.sin_d;

        self.k * (one + cos_term)
    }

    /// Computes dV/d(xi).
    ///
    /// ```text
    /// dV/d(xi) = -k * n * sin(n*xi - d)
    ///          = -k * n * (sin(n*xi)*cos(d) - cos(n*xi)*sin(d))
    /// ```
    #[inline(always)]
    fn derivative(&self, cos_xi: T, sin_xi: T) -> T {
        let (cos_n, sin_n) = chebyshev_cos_sin(self.n, cos_xi, sin_xi);

        // sin(n*xi - d) = sin(n*xi)*cos(d) - cos(n*xi)*sin(d)
        let sin_term = sin_n * self.cos_d - cos_n * self.sin_d;

        // dV/dxi = -k * n * sin(n*xi - d)
        self.k * self.neg_n * sin_term
    }

    /// Computes energy and derivative together (optimized).
    ///
    /// Shares the computation of cos(n*xi) and sin(n*xi) via Chebyshev.
    #[inline(always)]
    fn energy_derivative(&self, cos_xi: T, sin_xi: T) -> (T, T) {
        let one = T::splat(1.0);

        let (cos_n, sin_n) = chebyshev_cos_sin(self.n, cos_xi, sin_xi);

        let cos_term = cos_n * self.cos_d + sin_n * self.sin_d;
        let sin_term = sin_n * self.cos_d - cos_n * self.sin_d;

        let energy = self.k * (one + cos_term);
        let derivative = self.k * self.neg_n * sin_term;

        (energy, derivative)
    }
}

/// Computes cos(n*x) and sin(n*x) using Chebyshev recursion.
///
/// Uses:
/// - cos((k+1)*x) = 2*cos(x)*cos(k*x) - cos((k-1)*x)
/// - sin((k+1)*x) = 2*cos(x)*sin(k*x) - sin((k-1)*x)
#[inline(always)]
fn chebyshev_cos_sin<T: Vector>(n: i32, cos_x: T, sin_x: T) -> (T, T) {
    let zero = T::zero();
    let one = T::splat(1.0);
    let two = T::splat(2.0);

    match n {
        0 => (one, zero),
        1 => (cos_x, sin_x),
        2 => {
            // cos(2x) = 2*cos^2(x) - 1
            // sin(2x) = 2*sin(x)*cos(x)
            let cos2 = two * cos_x * cos_x - one;
            let sin2 = two * sin_x * cos_x;
            (cos2, sin2)
        }
        3 => {
            // cos(3x) = 4*cos^3(x) - 3*cos(x)
            // sin(3x) = sin(x)*(4*cos^2(x) - 1)
            let four = T::splat(4.0);
            let three = T::splat(3.0);
            let cos2 = cos_x * cos_x;
            let cos3 = four * cos2 * cos_x - three * cos_x;
            let sin3 = sin_x * (four * cos2 - one);
            (cos3, sin3)
        }
        _ => {
            // General Chebyshev recursion
            let mut cos_prev = one;
            let mut sin_prev = zero;
            let mut cos_curr = cos_x;
            let mut sin_curr = sin_x;

            for _ in 1..n {
                let cos_next = two * cos_x * cos_curr - cos_prev;
                let sin_next = two * cos_x * sin_curr - sin_prev;
                cos_prev = cos_curr;
                sin_prev = sin_curr;
                cos_curr = cos_next;
                sin_curr = sin_next;
            }

            (cos_curr, sin_curr)
        }
    }
}
