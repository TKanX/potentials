//! # Physics Interface Traits
//!
//! Defines the core trait abstractions for potential energy computations.
//! Different interaction types have specialized signatures optimized for
//! their specific computational patterns.
//!
//! ## Trait Hierarchy
//!
//! - [`Potential2`]: 2-body interactions (pairs, bonds)
//! - [`Potential3`]: 3-body interactions (angles)
//! - [`Potential4`]: 4-body interactions (torsions, impropers)
//!
//! ## Design Philosophy
//!
//! ### Raw Derivatives
//!
//! Potentials return "raw" derivatives rather than Cartesian forces:
//!
//! - **2-Body**: Returns force factor `S = -(dV/dr) / r`
//!   - Actual force: `F_vec = S * r_vec`
//!
//! - **3-Body**: Returns `dV/d(cos_theta)`
//!   - Caller applies chain rule for Cartesian mapping
//!
//! - **4-Body**: Returns `dV/d(phi)`
//!   - Caller applies chain rule for Cartesian mapping
//!
//! This separation keeps potential implementations simple and pure mathematical,
//! while allowing the caller (force field engine) to handle coordinate transformations.

use crate::math::Vector;

// ============================================================================
// 2-Body Potential (Pair & Bond)
// ============================================================================

/// Two-body potential energy function.
///
/// Used for both non-bonded pair interactions (LJ, Coulomb) and
/// bonded stretch interactions (harmonic, Morse).
///
/// ## Input Convention
///
/// All methods accept `r_sq` (squared distance) rather than `r` to avoid
/// unnecessary square root operations in the common case.
///
/// ## Force Factor Convention
///
/// The `force_factor` method returns:
///
/// ```text
/// S = -(dV/dr) / r = -dV/d(r^2) * 2
/// ```
///
/// To obtain the force vector from particle j to particle i:
///
/// ```text
/// F_ij = S * r_ij_vec
/// ```
///
/// where `r_ij_vec = r_i - r_j` (vector from j to i).
pub trait Potential2<T: Vector> {
    /// Computes the potential energy.
    ///
    /// ## Arguments
    ///
    /// - `r_sq`: Squared distance between particles (r^2)
    ///
    /// ## Returns
    ///
    /// Potential energy V(r)
    fn energy(&self, r_sq: T) -> T;

    /// Computes the force factor.
    ///
    /// ## Arguments
    ///
    /// - `r_sq`: Squared distance between particles (r^2)
    ///
    /// ## Returns
    ///
    /// Force factor `S = -(dV/dr) / r`
    ///
    /// The actual force vector is: `F = S * r_vec`
    fn force_factor(&self, r_sq: T) -> T;

    /// Computes both energy and force factor simultaneously.
    ///
    /// Default implementation calls both methods separately.
    /// Override for potentials with shared subexpressions.
    ///
    /// ## Returns
    ///
    /// Tuple of (energy, force_factor)
    #[inline(always)]
    fn energy_force(&self, r_sq: T) -> (T, T) {
        (self.energy(r_sq), self.force_factor(r_sq))
    }
}

// ============================================================================
// 3-Body Potential (Angle)
// ============================================================================

/// Three-body angular potential energy function.
///
/// Computes energy as a function of the angle formed by three particles (i-j-k),
/// where j is the central vertex.
///
/// ## Input Convention
///
/// Methods accept:
/// - `r_ij_sq`: Squared distance i-j
/// - `r_jk_sq`: Squared distance j-k
/// - `cos_theta`: Cosine of angle i-j-k (computed from dot product)
///
/// The cosine is preferred over the angle itself because:
/// 1. It's directly available from the dot product: `cos_theta = (r_ij · r_jk) / (|r_ij| |r_jk|)`
/// 2. Many angle potentials (GROMOS, DREIDING) use cosine directly
/// 3. Avoids expensive `acos` operation in the common case
///
/// ## Derivative Convention
///
/// The `derivative` method returns `dV/d(cos_theta)`.
///
/// The caller is responsible for applying the chain rule to obtain
/// Cartesian forces. For angle i-j-k:
///
/// ```text
/// d(cos_theta)/d(r_i) = (r_jk/|r_jk| - cos_theta * r_ij/|r_ij|) / |r_ij|
/// ```
///
/// (and similar expressions for j and k)
pub trait Potential3<T: Vector> {
    /// Computes the potential energy.
    ///
    /// ## Arguments
    ///
    /// - `r_ij_sq`: Squared distance from i to j
    /// - `r_jk_sq`: Squared distance from j to k
    /// - `cos_theta`: Cosine of angle i-j-k
    ///
    /// ## Returns
    ///
    /// Potential energy V(theta)
    fn energy(&self, r_ij_sq: T, r_jk_sq: T, cos_theta: T) -> T;

    /// Computes the derivative with respect to cos(theta).
    ///
    /// ## Arguments
    ///
    /// - `r_ij_sq`: Squared distance from i to j
    /// - `r_jk_sq`: Squared distance from j to k
    /// - `cos_theta`: Cosine of angle i-j-k
    ///
    /// ## Returns
    ///
    /// Derivative `dV/d(cos_theta)`
    fn derivative(&self, r_ij_sq: T, r_jk_sq: T, cos_theta: T) -> T;

    /// Computes both energy and derivative simultaneously.
    ///
    /// ## Returns
    ///
    /// Tuple of (energy, dV/d(cos_theta))
    #[inline(always)]
    fn energy_derivative(&self, r_ij_sq: T, r_jk_sq: T, cos_theta: T) -> (T, T) {
        (
            self.energy(r_ij_sq, r_jk_sq, cos_theta),
            self.derivative(r_ij_sq, r_jk_sq, cos_theta),
        )
    }
}
