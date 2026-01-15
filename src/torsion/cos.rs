//! # Periodic Cosine Torsion Potential
//!
//! The standard periodic torsion potential used by most force fields.
//!
//! ## Formula
//!
//! ```text
//! V(phi) = k * (1 + cos(n * phi - delta))
//! ```
//!
//! where:
//! - `k`: Barrier height (energy units)
//! - `n`: Periodicity (multiplicity)
//! - `delta`: Phase shift (radians)
//!
//! ## Derivative
//!
//! ```text
//! dV/d(phi) = -k * n * sin(n*phi - delta)
//! ```
//!
//! ## Implementation Notes
//!
//! - Uses Chebyshev recursion for efficient cos(n*phi) computation
//! - Common periodicities: n=1 (amide), n=3 (sp3 carbon), n=6 (aromatic)

use crate::base::Potential4;
use crate::math::Vector;

/// Periodic cosine torsion potential.
///
/// ## Parameters
///
/// - `k`: Barrier height (energy units)
/// - `n`: Periodicity (integer)
/// - `delta`: Phase shift (radians)
///
/// ## Precomputed Values
///
/// - `cos_delta`, `sin_delta`: Precomputed for angle-subtraction formula
#[derive(Clone, Copy, Debug)]
pub struct Cos<T> {
    /// Barrier height
    k: T,
    /// Periodicity
    n: i32,
    /// Cosine of phase shift
    cos_delta: T,
    /// Sine of phase shift
    sin_delta: T,
}

impl<T: Vector> Cos<T> {
    /// Creates a new periodic cosine torsion.
    ///
    /// ## Arguments
    ///
    /// - `k`: Barrier height (energy units)
    /// - `n`: Periodicity (1, 2, 3, etc.)
    /// - `delta`: Phase shift in radians
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::torsion::Cos;
    ///
    /// // Ethane-like rotation: 3-fold, cis minimum
    /// let torsion = Cos::<f64>::new(2.5, 3, 0.0);
    /// ```
    #[inline]
    pub fn new(k: f64, n: i32, delta: f64) -> Self {
        Self {
            k: T::splat(k),
            n,
            cos_delta: T::splat(delta.cos()),
            sin_delta: T::splat(delta.sin()),
        }
    }

    /// Creates with delta in degrees.
    #[inline]
    pub fn from_degrees(k: f64, n: i32, delta_deg: f64) -> Self {
        let delta_rad = delta_deg * core::f64::consts::PI / 180.0;
        Self::new(k, n, delta_rad)
    }

    /// Returns the barrier height.
    #[inline]
    pub fn k(&self) -> T {
        self.k
    }

    /// Returns the periodicity.
    #[inline]
    pub fn n(&self) -> i32 {
        self.n
    }

    /// Computes cos(n*phi) using Chebyshev recursion.
    ///
    /// Uses T_n(cos(phi)) = cos(n*phi) with recursion:
    /// T_0 = 1, T_1 = x, T_n = 2*x*T_{n-1} - T_{n-2}
    #[inline(always)]
    fn cos_n_phi(&self, cos_phi: T) -> T {
        match self.n {
            0 => T::one(),
            1 => cos_phi,
            2 => {
                // cos(2*phi) = 2*cos^2(phi) - 1
                let two = T::splat(2.0);
                two * cos_phi * cos_phi - T::one()
            }
            3 => {
                // cos(3*phi) = 4*cos^3(phi) - 3*cos(phi)
                let three = T::splat(3.0);
                let four = T::splat(4.0);
                four * cos_phi * cos_phi * cos_phi - three * cos_phi
            }
            _ => {
                // General Chebyshev recursion
                let two = T::splat(2.0);
                let mut t_prev2 = T::one();
                let mut t_prev1 = cos_phi;

                for _ in 2..=self.n.unsigned_abs() {
                    let t_curr = two * cos_phi * t_prev1 - t_prev2;
                    t_prev2 = t_prev1;
                    t_prev1 = t_curr;
                }

                // cos(-n*phi) = cos(n*phi), so sign of n doesn't matter
                t_prev1
            }
        }
    }

    /// Computes sin(n*phi) using Chebyshev U polynomials.
    ///
    /// sin(n*phi) = sin(phi) * U_{n-1}(cos(phi))
    #[inline(always)]
    fn sin_n_phi(&self, cos_phi: T, sin_phi: T) -> T {
        match self.n {
            0 => T::zero(),
            1 => sin_phi,
            2 => {
                // sin(2*phi) = 2*sin(phi)*cos(phi)
                let two = T::splat(2.0);
                two * sin_phi * cos_phi
            }
            3 => {
                // sin(3*phi) = sin(phi) * (4*cos^2(phi) - 1)
                let four = T::splat(4.0);
                sin_phi * (four * cos_phi * cos_phi - T::one())
            }
            _ => {
                // General recursion for U_n: U_0 = 1, U_1 = 2x, U_n = 2x*U_{n-1} - U_{n-2}
                let two = T::splat(2.0);
                let n_abs = self.n.unsigned_abs() as i32;

                if n_abs == 0 {
                    return T::zero();
                }

                let mut u_prev2 = T::one();
                let mut u_prev1 = two * cos_phi;

                if n_abs == 1 {
                    return sin_phi;
                }

                for _ in 2..n_abs {
                    let u_curr = two * cos_phi * u_prev1 - u_prev2;
                    u_prev2 = u_prev1;
                    u_prev1 = u_curr;
                }

                let result = sin_phi * u_prev1;
                if self.n < 0 {
                    T::zero() - result // sin(-n*phi) = -sin(n*phi)
                } else {
                    result
                }
            }
        }
    }
}

impl<T: Vector> Potential4<T> for Cos<T> {
    /// Computes the potential energy.
    ///
    /// ```text
    /// V = k * (1 + cos(n*phi - delta))
    ///   = k * (1 + cos(n*phi)*cos(delta) + sin(n*phi)*sin(delta))
    /// ```
    #[inline(always)]
    fn energy(&self, cos_phi: T, sin_phi: T) -> T {
        let cos_n = self.cos_n_phi(cos_phi);
        let sin_n = self.sin_n_phi(cos_phi, sin_phi);

        // cos(n*phi - delta) = cos(n*phi)*cos(delta) + sin(n*phi)*sin(delta)
        let cos_term = cos_n * self.cos_delta + sin_n * self.sin_delta;

        self.k * (T::one() + cos_term)
    }

    /// Computes dV/d(phi).
    ///
    /// ```text
    /// dV/d(phi) = -k * n * sin(n*phi - delta)
    ///           = -k * n * (sin(n*phi)*cos(delta) - cos(n*phi)*sin(delta))
    /// ```
    #[inline(always)]
    fn derivative(&self, cos_phi: T, sin_phi: T) -> T {
        let cos_n = self.cos_n_phi(cos_phi);
        let sin_n = self.sin_n_phi(cos_phi, sin_phi);

        // sin(n*phi - delta) = sin(n*phi)*cos(delta) - cos(n*phi)*sin(delta)
        let sin_term = sin_n * self.cos_delta - cos_n * self.sin_delta;

        let n_t = T::splat(self.n as f64);
        T::zero() - self.k * n_t * sin_term
    }

    /// Computes energy and derivative together (optimized).
    ///
    /// Shares the computation of cos(n*phi) and sin(n*phi).
    #[inline(always)]
    fn energy_derivative(&self, cos_phi: T, sin_phi: T) -> (T, T) {
        let cos_n = self.cos_n_phi(cos_phi);
        let sin_n = self.sin_n_phi(cos_phi, sin_phi);

        let cos_term = cos_n * self.cos_delta + sin_n * self.sin_delta;
        let sin_term = sin_n * self.cos_delta - cos_n * self.sin_delta;

        let energy = self.k * (T::one() + cos_term);

        let n_t = T::splat(self.n as f64);
        let derivative = T::zero() - self.k * n_t * sin_term;

        (energy, derivative)
    }
}
