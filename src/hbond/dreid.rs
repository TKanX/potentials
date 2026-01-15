//! # DREIDING Hydrogen Bond Potential
//!
//! The 12-10 hydrogen bond potential from the DREIDING force field.
//!
//! ## Formula
//!
//! ```text
//! V(R, cos_theta) = D0 * [5*(R0/R)^12 - 6*(R0/R)^10] * cos^N(theta)
//! ```
//!
//! where:
//! - `R`: D···A distance (donor to acceptor, **not** H···A)
//! - `cos_theta`: Cosine of the D-H···A angle (passed directly, not the angle)
//! - `D0`: Well depth (energy)
//! - `R0`: Equilibrium D···A distance (length, typically ~2.75 Å)
//! - `N`: Angular exponent (4 in original DREIDING, 2 in some variants)
//!
//! ## Important: Distance Definition
//!
//! The distance parameter is the **donor-acceptor** distance, not the
//! hydrogen-acceptor distance. For O-H···O hydrogen bonds:
//! - D···A (O···O) distance: ~2.7-2.9 Å (use this!)
//! - H···A (H···O) distance: ~1.8-2.0 Å (do NOT use)
//!
//! Using H···A distance with DREIDING parameters will give incorrect results.
//!
//! ## Important: Angular Exponent Parity
//!
//! The code computes `cos_theta.powi(N)` directly. This has implications:
//!
//! - **Even N** (2, 4, 6): Maximum at `cos_theta = ±1`. Standard H-bond geometry
//!   (θ = 180°, cos = -1) produces maximum attraction. **Use even N for H-bonds.**
//! - **Odd N** (1, 3, 5): Maximum only at `cos_theta = +1`. If your geometry
//!   gives cos = -1 at linear, you'll get repulsion instead of attraction.
//!
//! DREIDING uses N=4 (even), so this is mathematically correct for the intended use.
//!
//! ## Properties
//!
//! - Minimum energy: -D0 at R=R0, cos_theta=±1 (for even N)
//! - Angular dependence: cos^N modulates radial interaction
//! - 12-10 form: softer repulsion than standard 12-6 LJ

use crate::math::Vector;

/// DREIDING hydrogen bond potential (12-10 with angular modulation).
///
/// ## Type Parameters
///
/// - `T`: Numeric type (f32, f64, or SIMD vector)
/// - `N`: Angular exponent (power of cos, compile-time constant)
///
/// ## Parameters
///
/// - `d0`: Well depth (energy)
/// - `r0`: Equilibrium D···A distance (length, **not** H···A)
///
/// ## Usage
///
/// This potential requires both distance and angle inputs.
/// The distance must be the **donor-acceptor** distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dreid<T, const N: u32 = 4> {
    d0: T,
    r0_sq: T,
    neg_60_d0: T,
}

impl<T: Vector, const N: u32> Dreid<T, N> {
    /// Creates a DREIDING hydrogen bond potential.
    ///
    /// ## Arguments
    ///
    /// - `d0`: Well depth (energy)
    /// - `r0`: Equilibrium D···A distance (length)
    ///
    /// ## Example
    ///
    /// ```
    /// use potentials::hbond::Dreid;
    ///
    /// // O-H···O hydrogen bond with DREIDING parameters
    /// // D0 = 8 kcal/mol, R0 = 2.75 Å (O···O distance!)
    /// let hbond = Dreid::<f64, 4>::new(8.0, 2.75);  // cos^4 (original)
    /// let hbond2 = Dreid::<f64, 2>::new(8.0, 2.75); // cos^2 (variant)
    /// ```
    #[inline]
    pub fn new(d0: f64, r0: f64) -> Self {
        Self {
            d0: T::splat(d0),
            r0_sq: T::splat(r0 * r0),
            neg_60_d0: T::splat(-60.0 * d0),
        }
    }

    /// Computes the full potential energy.
    ///
    /// ## Arguments
    ///
    /// - `r_sq`: Squared D···A distance (length²)
    /// - `cos_theta`: cos(D-H···A angle)
    #[inline(always)]
    pub fn energy(&self, r_sq: T, cos_theta: T) -> T {
        let five = T::splat(5.0);
        let six = T::splat(6.0);

        // (R0/R)^2
        let ratio2 = self.r0_sq / r_sq;
        let ratio4 = ratio2 * ratio2;
        let ratio8 = ratio4 * ratio4;
        let ratio10 = ratio8 * ratio2;
        let ratio12 = ratio10 * ratio2;

        // cos^N(theta) - optimized at compile time
        let cos_n = cos_power::<T, N>(cos_theta);

        // D0 * (5*(R0/R)^12 - 6*(R0/R)^10) * cos^N
        self.d0 * (five * ratio12 - six * ratio10) * cos_n
    }

    /// Computes radial and angular derivatives.
    ///
    /// ## Returns
    ///
    /// - `(S, dV_dcos)` where:
    ///   - `S = -(dV/dr)/r` for computing forces from distance vector
    ///   - `dV_dcos` for computing angular forces
    #[inline(always)]
    pub fn derivative(&self, r_sq: T, cos_theta: T) -> (T, T) {
        let five = T::splat(5.0);
        let six = T::splat(6.0);

        // Powers of (R0/R)^2
        let ratio2 = self.r0_sq / r_sq;
        let ratio4 = ratio2 * ratio2;
        let ratio8 = ratio4 * ratio4;
        let ratio10 = ratio8 * ratio2;
        let ratio12 = ratio10 * ratio2;

        // Angular terms - optimized at compile time
        let cos_n = cos_power::<T, N>(cos_theta);
        let cos_nm1 = cos_power_m1::<T, N>(cos_theta);

        // Radial derivative: dV/dr
        // S = -(dV/dr)/r = 60 * D0 * ((R0/R)^12 - (R0/R)^10) * cos^N / r^2
        let s = self.neg_60_d0 * (ratio10 - ratio12) * cos_n / r_sq;

        // Angular derivative: dV/d(cos_theta)
        // dV/d(cos) = D0 * (5*ratio12 - 6*ratio10) * N * cos^(N-1)
        let n_t = T::splat(N as f64);
        let radial_part = five * ratio12 - six * ratio10;
        let dv_dcos = self.d0 * radial_part * n_t * cos_nm1;

        (s, dv_dcos)
    }

    /// Computes energy and both derivatives together (optimized).
    ///
    /// Shares computations for efficiency.
    #[inline(always)]
    pub fn energy_derivative(&self, r_sq: T, cos_theta: T) -> (T, T, T) {
        let five = T::splat(5.0);
        let six = T::splat(6.0);

        let ratio2 = self.r0_sq / r_sq;
        let ratio4 = ratio2 * ratio2;
        let ratio8 = ratio4 * ratio4;
        let ratio10 = ratio8 * ratio2;
        let ratio12 = ratio10 * ratio2;

        let cos_n = cos_power::<T, N>(cos_theta);
        let cos_nm1 = cos_power_m1::<T, N>(cos_theta);

        let radial_part = five * ratio12 - six * ratio10;

        let energy = self.d0 * radial_part * cos_n;
        let s = self.neg_60_d0 * (ratio10 - ratio12) * cos_n / r_sq;
        let n_t = T::splat(N as f64);
        let dv_dcos = self.d0 * radial_part * n_t * cos_nm1;

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
        0 | 1 => T::splat(1.0), // cos^0 = 1, cos^(-1) not used (N=0 gives dV/dcos=0)
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
            // Binary exponentiation for large N
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
