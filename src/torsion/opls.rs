//! # OPLS Torsion Potential (Fourier Series)
//!
//! A Fourier series expansion commonly used in OPLS and CHARMM force fields.
//!
//! ## Formula
//!
//! ```text
//! V(phi) = c1 * (1 + cos(phi))
//!        + c2 * (1 - cos(2*phi))
//!        + c3 * (1 + cos(3*phi))
//!        + c4 * (1 - cos(4*phi))
//! ```
//!
//! or equivalently:
//!
//! ```text
//! V(phi) = (c1 + c2 + c3 + c4)
//!        + c1*cos(phi) - c2*cos(2*phi) + c3*cos(3*phi) - c4*cos(4*phi)
//! ```
//!
//! ## Parameters
//!
//! - `c1`: Coefficient for n=1 term (energy units)
//! - `c2`: Coefficient for n=2 term (energy units)
//! - `c3`: Coefficient for n=3 term (energy units)
//! - `c4`: Coefficient for n=4 term (energy units)
//!
//! ## Implementation Notes
//!
//! - Uses Chebyshev polynomials for efficient cos(n*phi) computation
//! - The alternating signs ensure minimum at phi=0 for positive coefficients
//! - c4 is often zero in simpler parameterizations

use crate::base::Potential4;
use crate::math::Vector;

/// OPLS Fourier series torsion potential.
///
/// ## Parameters
///
/// - `c1, c2, c3, c4`: Fourier coefficients (energy units)
///
/// ## Precomputed Values
///
/// - `offset`: Stores `c1 + c2 + c3 + c4` for efficient computation
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Opls<T> {
    /// Coefficient for n=1 term
    c1: T,
    /// Coefficient for n=2 term
    c2: T,
    /// Coefficient for n=3 term
    c3: T,
    /// Coefficient for n=4 term
    c4: T,
    /// Constant offset: c1 + c2 + c3 + c4
    offset: T,
}

impl<T: Vector> Opls<T> {
    /// Creates a new OPLS torsion potential.
    ///
    /// ## Arguments
    ///
    /// - `c1, c2, c3, c4`: Fourier coefficients
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::torsion::Opls;
    ///
    /// // Typical alkane torsion
    /// let torsion = Opls::<f64>::new(0.7, 0.1, 1.5, 0.0);
    /// ```
    #[inline]
    pub fn new(c1: f64, c2: f64, c3: f64, c4: f64) -> Self {
        Self {
            c1: T::splat(c1),
            c2: T::splat(c2),
            c3: T::splat(c3),
            c4: T::splat(c4),
            offset: T::splat(c1 + c2 + c3 + c4),
        }
    }

    /// Creates with only n=1 and n=3 terms (common case).
    #[inline]
    pub fn simple(c1: f64, c3: f64) -> Self {
        Self::new(c1, 0.0, c3, 0.0)
    }

    /// Creates with only n=3 term (pure three-fold).
    #[inline]
    pub fn threefold(c3: f64) -> Self {
        Self::new(0.0, 0.0, c3, 0.0)
    }
}

impl<T: Vector> Potential4<T> for Opls<T> {
    /// Computes the potential energy.
    ///
    /// ```text
    /// V = offset + c1*cos(phi) - c2*cos(2phi) + c3*cos(3phi) - c4*cos(4phi)
    /// ```
    ///
    /// Uses Chebyshev polynomials for cos(n*phi):
    /// - cos(2phi) = 2*cos^2(phi) - 1
    /// - cos(3phi) = 4*cos^3(phi) - 3*cos(phi)
    /// - cos(4phi) = 8*cos^4(phi) - 8*cos^2(phi) + 1
    #[inline(always)]
    fn energy(&self, cos_phi: T, _sin_phi: T) -> T {
        let cos2 = cos_phi * cos_phi;
        let cos3 = cos2 * cos_phi;
        let cos4 = cos2 * cos2;

        let two = T::splat(2.0);
        let three = T::splat(3.0);
        let four = T::splat(4.0);
        let eight = T::splat(8.0);

        // cos(2phi) = 2*cos^2 - 1
        let cos_2phi = two * cos2 - T::one();

        // cos(3phi) = 4*cos^3 - 3*cos
        let cos_3phi = four * cos3 - three * cos_phi;

        // cos(4phi) = 8*cos^4 - 8*cos^2 + 1
        let cos_4phi = eight * cos4 - eight * cos2 + T::one();

        self.offset + self.c1 * cos_phi - self.c2 * cos_2phi + self.c3 * cos_3phi
            - self.c4 * cos_4phi
    }

    /// Computes dV/d(phi).
    ///
    /// ```text
    /// dV/dphi = -c1*sin(phi) + 2*c2*sin(2phi) - 3*c3*sin(3phi) + 4*c4*sin(4phi)
    /// ```
    #[inline(always)]
    fn derivative(&self, cos_phi: T, sin_phi: T) -> T {
        let cos2 = cos_phi * cos_phi;

        let two = T::splat(2.0);
        let three = T::splat(3.0);
        let four = T::splat(4.0);

        // sin(2phi) = 2*sin*cos
        let sin_2phi = two * sin_phi * cos_phi;

        // sin(3phi) = sin * (4*cos^2 - 1)
        let sin_3phi = sin_phi * (four * cos2 - T::one());

        // sin(4phi) = 4*sin*cos * (2*cos^2 - 1)
        let sin_4phi = four * sin_phi * cos_phi * (two * cos2 - T::one());

        T::zero() - self.c1 * sin_phi + two * self.c2 * sin_2phi - three * self.c3 * sin_3phi
            + four * self.c4 * sin_4phi
    }

    /// Computes energy and derivative together (optimized).
    ///
    /// Shares the computation of cos^n and sin(n*phi) terms.
    #[inline(always)]
    fn energy_derivative(&self, cos_phi: T, sin_phi: T) -> (T, T) {
        let cos2 = cos_phi * cos_phi;
        let cos3 = cos2 * cos_phi;
        let cos4 = cos2 * cos2;

        let two = T::splat(2.0);
        let three = T::splat(3.0);
        let four = T::splat(4.0);
        let eight = T::splat(8.0);

        // cos(n*phi) terms
        let cos_2phi = two * cos2 - T::one();
        let cos_3phi = four * cos3 - three * cos_phi;
        let cos_4phi = eight * cos4 - eight * cos2 + T::one();

        let energy = self.offset + self.c1 * cos_phi - self.c2 * cos_2phi + self.c3 * cos_3phi
            - self.c4 * cos_4phi;

        // sin(n*phi) terms
        let sin_2phi = two * sin_phi * cos_phi;
        let sin_3phi = sin_phi * (four * cos2 - T::one());
        let sin_4phi = four * sin_phi * cos_phi * (two * cos2 - T::one());

        let derivative = T::zero() - self.c1 * sin_phi + two * self.c2 * sin_2phi
            - three * self.c3 * sin_3phi
            + four * self.c4 * sin_4phi;

        (energy, derivative)
    }
}
