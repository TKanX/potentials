//! # Ryckaert-Bellemans Torsion Potential
//!
//! A polynomial expansion in cos(phi) used by GROMOS force fields.
//!
//! ## Formula
//!
//! ```text
//! V(phi) = sum_{n=0}^5 c_n * cos^n(phi)
//! ```
//!
//! Expanded:
//!
//! ```text
//! V = c0 + c1*cos(phi) + c2*cos^2(phi) + c3*cos^3(phi) + c4*cos^4(phi) + c5*cos^5(phi)
//! ```
//!
//! ## Derivative
//!
//! ```text
//! dV/d(phi) = -sin(phi) * sum_{n=1}^5 n * c_n * cos^{n-1}(phi)
//! ```
//!
//! ## Implementation Notes
//!
//! - Uses Horner's method for efficient polynomial evaluation
//! - Only requires powers of cos(phi), no trigonometric calls needed
//! - Common in GROMOS force field (c0-c5 coefficients)
//! - phi = 0 corresponds to cis configuration

use crate::base::Potential4;
use crate::math::Vector;

/// Ryckaert-Bellemans torsion potential.
///
/// ## Parameters
///
/// - `c0` through `c5`: Polynomial coefficients (energy units)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rb<T> {
    c0: T,
    c1: T,
    c2: T,
    c3: T,
    c4: T,
    c5: T,
}

impl<T: Vector> Rb<T> {
    /// Creates a new Ryckaert-Bellemans potential.
    ///
    /// ## Arguments
    ///
    /// - `c0` through `c5`: Polynomial coefficients
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::torsion::Rb;
    ///
    /// // Typical alkane torsion (GROMOS)
    /// let torsion = Rb::<f64>::new(9.28, 12.16, -13.12, -3.06, 26.24, 0.0);
    /// ```
    #[inline]
    pub fn new(c0: f64, c1: f64, c2: f64, c3: f64, c4: f64, c5: f64) -> Self {
        Self {
            c0: T::splat(c0),
            c1: T::splat(c1),
            c2: T::splat(c2),
            c3: T::splat(c3),
            c4: T::splat(c4),
            c5: T::splat(c5),
        }
    }

    /// Creates from array of coefficients.
    #[inline]
    pub fn from_array(c: [f64; 6]) -> Self {
        Self::new(c[0], c[1], c[2], c[3], c[4], c[5])
    }

    /// Creates with only c0-c3 (common 4-term form).
    #[inline]
    pub fn four_term(c0: f64, c1: f64, c2: f64, c3: f64) -> Self {
        Self::new(c0, c1, c2, c3, 0.0, 0.0)
    }
}

impl<T: Vector> Potential4<T> for Rb<T> {
    /// Computes the potential energy using Horner's method.
    ///
    /// ```text
    /// V = c0 + cos * (c1 + cos * (c2 + cos * (c3 + cos * (c4 + cos * c5))))
    /// ```
    #[inline(always)]
    fn energy(&self, cos_phi: T, _sin_phi: T) -> T {
        // Horner's method for efficiency
        let result = self.c5;
        let result = result * cos_phi + self.c4;
        let result = result * cos_phi + self.c3;
        let result = result * cos_phi + self.c2;
        let result = result * cos_phi + self.c1;
        result * cos_phi + self.c0
    }

    /// Computes dV/d(phi).
    ///
    /// ```text
    /// dV/d(cos) = c1 + 2*c2*cos + 3*c3*cos^2 + 4*c4*cos^3 + 5*c5*cos^4
    /// dV/d(phi) = dV/d(cos) * d(cos)/d(phi) = -dV/d(cos) * sin(phi)
    /// ```
    #[inline(always)]
    fn derivative(&self, cos_phi: T, sin_phi: T) -> T {
        // dV/d(cos) using Horner's method
        let five = T::splat(5.0);
        let four = T::splat(4.0);
        let three = T::splat(3.0);
        let two = T::splat(2.0);

        let dv_dcos = five * self.c5;
        let dv_dcos = dv_dcos * cos_phi + four * self.c4;
        let dv_dcos = dv_dcos * cos_phi + three * self.c3;
        let dv_dcos = dv_dcos * cos_phi + two * self.c2;
        let dv_dcos = dv_dcos * cos_phi + self.c1;

        // dV/dphi = -dV/dcos * sin(phi)
        T::zero() - dv_dcos * sin_phi
    }

    /// Computes energy and derivative together (optimized).
    ///
    /// Shares the computation of powers of cos(phi).
    #[inline(always)]
    fn energy_derivative(&self, cos_phi: T, sin_phi: T) -> (T, T) {
        let five = T::splat(5.0);
        let four = T::splat(4.0);
        let three = T::splat(3.0);
        let two = T::splat(2.0);

        // Build up powers of cos and accumulate both energy and derivative
        let cos2 = cos_phi * cos_phi;
        let cos3 = cos2 * cos_phi;
        let cos4 = cos2 * cos2;
        let cos5 = cos4 * cos_phi;

        let energy = self.c0
            + self.c1 * cos_phi
            + self.c2 * cos2
            + self.c3 * cos3
            + self.c4 * cos4
            + self.c5 * cos5;

        let dv_dcos = self.c1
            + two * self.c2 * cos_phi
            + three * self.c3 * cos2
            + four * self.c4 * cos3
            + five * self.c5 * cos4;

        let derivative = T::zero() - dv_dcos * sin_phi;

        (energy, derivative)
    }
}
