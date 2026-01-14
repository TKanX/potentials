//! # Math Abstraction Layer
//!
//! Provides a unified interface for scalar floating-point operations (`f32`, `f64`).
//! All potential computations are generic over `T: Vector`, enabling a single codebase
//! to support both single and double precision.
//!
//! ## Design Philosophy
//!
//! - **Branchless**: All conditionals use masks and select operations
//! - **Zero-Cost Abstraction**: Thin wrappers with no runtime overhead
//! - **Platform Independent**: Works in both `std` and `no_std` environments
//!
//! ## Feature Flags
//!
//! - `std` (default): Uses standard library math functions
//! - `libm`: Uses `libm` crate for `no_std` environments
//!
//! ## Supported Types
//!
//! | Vector Type | Mask Type | Description |
//! |-------------|-----------|-------------|
//! | `f32` | [`MaskF32`] | 32-bit floating point |
//! | `f64` | [`MaskF64`] | 64-bit floating point |

use core::ops::{Add, BitAnd, BitOr, Div, Mul, Neg, Not, Sub};

// ============================================================================
// Mask Trait
// ============================================================================

/// Boolean mask abstraction for branchless conditional operations.
///
/// Masks represent lane-wise boolean values that can be combined with
/// logical operations and used to select between values.
pub trait Mask:
    Copy + Clone + Sized + BitAnd<Output = Self> + BitOr<Output = Self> + Not<Output = Self>
{
    /// The vector type this mask is associated with.
    type Vector: Vector<Mask = Self>;

    /// Returns `true` if any lane is set.
    fn any(self) -> bool;

    /// Returns `true` if all lanes are set.
    fn all(self) -> bool;

    /// Branchless conditional select.
    ///
    /// Returns `if_true` where mask is set, `if_false` otherwise.
    fn select(self, if_true: Self::Vector, if_false: Self::Vector) -> Self::Vector;
}

// ============================================================================
// Float Trait (for scalar type abstraction)
// ============================================================================

/// Trait for floating-point scalar types (`f32` and `f64`).
///
/// This trait provides a unified interface for constants and conversions
/// between different floating-point precisions.
pub trait Float: Copy + Clone + Sized + PartialOrd {
    /// The mathematical constant π.
    const PI: Self;

    /// The mathematical constant e (Euler's number).
    const E: Self;

    /// Converts from `f64` to this type.
    fn from_f64(value: f64) -> Self;

    /// Converts this type to `f64`.
    fn to_f64(self) -> f64;
}

impl Float for f32 {
    const PI: Self = core::f32::consts::PI;
    const E: Self = core::f32::consts::E;

    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        value as f32
    }

    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl Float for f64 {
    const PI: Self = core::f64::consts::PI;
    const E: Self = core::f64::consts::E;

    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        value
    }

    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
}

// ============================================================================
// Vector Trait
// ============================================================================

/// Unified interface for scalar floating-point math operations.
///
/// This trait abstracts over scalar types (`f32`, `f64`),
/// providing a consistent API for all potential computations.
///
/// ## Required Operations
///
/// - **Arithmetic**: `+`, `-`, `*`, `/`, negation
/// - **Math**: `sqrt`, `rsqrt`, `recip`, `exp`, `ln`, `sin`, `cos`, `acos`, `powi`
/// - **Comparison**: `lt`, `le`, `gt`, `ge`, `eq`
/// - **Selection**: `select` for branchless conditionals
pub trait Vector:
    Copy
    + Clone
    + Sized
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Associated mask type for conditional operations.
    type Mask: Mask<Vector = Self>;

    /// The underlying scalar type for this vector.
    type Scalar: Float;

    /// Number of lanes in this vector (1 for scalars).
    const LANES: usize;

    // ========================================================================
    // Constants
    // ========================================================================

    /// Broadcasts a scalar value to all lanes.
    fn splat(value: f64) -> Self;

    /// Returns a vector of zeros.
    #[inline(always)]
    fn zero() -> Self {
        Self::splat(0.0)
    }

    /// Returns a vector of ones.
    #[inline(always)]
    fn one() -> Self {
        Self::splat(1.0)
    }

    // ========================================================================
    // Basic Math
    // ========================================================================

    /// Square root: `sqrt(x)`.
    fn sqrt(self) -> Self;

    /// Reciprocal square root: `1 / sqrt(x)`.
    #[inline(always)]
    fn rsqrt(self) -> Self {
        Self::one() / self.sqrt()
    }

    /// Reciprocal: `1 / x`.
    #[inline(always)]
    fn recip(self) -> Self {
        Self::one() / self
    }

    /// Absolute value: `|x|`.
    fn abs(self) -> Self;

    /// Minimum of two values (lane-wise).
    fn min(self, other: Self) -> Self;

    /// Maximum of two values (lane-wise).
    fn max(self, other: Self) -> Self;

    // ========================================================================
    // Transcendental Functions
    // ========================================================================

    /// Exponential: `e^x`.
    fn exp(self) -> Self;

    /// Natural logarithm: `ln(x)`.
    fn ln(self) -> Self;

    /// Sine: `sin(x)`.
    fn sin(self) -> Self;

    /// Cosine: `cos(x)`.
    fn cos(self) -> Self;

    /// Arc cosine: `acos(x)`.
    fn acos(self) -> Self;

    /// Arc sine: `asin(x)`.
    ///
    /// **Warning**: Expensive operation. Avoid in inner loops when possible.
    fn asin(self) -> Self;

    /// Two-argument arc tangent: `atan2(y, x)`.
    ///
    /// Returns the angle between the positive x-axis and the point (x, y).
    fn atan2(self, other: Self) -> Self;

    /// Integer power: `x^n`.
    fn powi(self, n: i32) -> Self;

    /// Floating-point power: `x^y`.
    fn powf(self, exp: Self) -> Self;

    // ========================================================================
    // Comparison (Returns Mask)
    // ========================================================================

    /// Less than: `x < y`.
    fn lt(self, other: Self) -> Self::Mask;

    /// Less than or equal: `x <= y`.
    fn le(self, other: Self) -> Self::Mask;

    /// Greater than: `x > y`.
    fn gt(self, other: Self) -> Self::Mask;

    /// Greater than or equal: `x >= y`.
    fn ge(self, other: Self) -> Self::Mask;

    /// Equal: `x == y` (exact floating-point equality).
    fn eq(self, other: Self) -> Self::Mask;
}

// ============================================================================
// Scalar Math Functions (std vs libm)
// ============================================================================

/// Scalar f32 math operations.
///
/// Provides platform-independent math functions for `f32` type.
/// Uses `std` library when available, falls back to `libm` in `no_std`.
pub(crate) mod scalar_f32 {
    #[cfg(feature = "std")]
    extern crate std;

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn sqrt(x: f32) -> f32 {
        x.sqrt()
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn sqrt(x: f32) -> f32 {
        libm::sqrtf(x)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn abs(x: f32) -> f32 {
        x.abs()
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn abs(x: f32) -> f32 {
        libm::fabsf(x)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn min(x: f32, y: f32) -> f32 {
        x.min(y)
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn min(x: f32, y: f32) -> f32 {
        libm::fminf(x, y)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn max(x: f32, y: f32) -> f32 {
        x.max(y)
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn max(x: f32, y: f32) -> f32 {
        libm::fmaxf(x, y)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn exp(x: f32) -> f32 {
        x.exp()
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn exp(x: f32) -> f32 {
        libm::expf(x)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn ln(x: f32) -> f32 {
        x.ln()
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn ln(x: f32) -> f32 {
        libm::logf(x)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn sin(x: f32) -> f32 {
        x.sin()
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn sin(x: f32) -> f32 {
        libm::sinf(x)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn cos(x: f32) -> f32 {
        x.cos()
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn cos(x: f32) -> f32 {
        libm::cosf(x)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn acos(x: f32) -> f32 {
        x.acos()
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn acos(x: f32) -> f32 {
        libm::acosf(x)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn asin(x: f32) -> f32 {
        x.asin()
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn asin(x: f32) -> f32 {
        libm::asinf(x)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn atan2(y: f32, x: f32) -> f32 {
        y.atan2(x)
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn atan2(y: f32, x: f32) -> f32 {
        libm::atan2f(y, x)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn powi(x: f32, n: i32) -> f32 {
        x.powi(n)
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn powi(x: f32, n: i32) -> f32 {
        libm::powf(x, n as f32)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn powf(x: f32, y: f32) -> f32 {
        x.powf(y)
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn powf(x: f32, y: f32) -> f32 {
        libm::powf(x, y)
    }
}

/// Scalar f64 math operations.
///
/// Provides platform-independent math functions for `f64` type.
/// Uses `std` library when available, falls back to `libm` in `no_std`.
pub(crate) mod scalar_f64 {
    #[cfg(feature = "std")]
    extern crate std;

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn sqrt(x: f64) -> f64 {
        x.sqrt()
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn sqrt(x: f64) -> f64 {
        libm::sqrt(x)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn abs(x: f64) -> f64 {
        x.abs()
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn abs(x: f64) -> f64 {
        libm::fabs(x)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn min(x: f64, y: f64) -> f64 {
        x.min(y)
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn min(x: f64, y: f64) -> f64 {
        libm::fmin(x, y)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn max(x: f64, y: f64) -> f64 {
        x.max(y)
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn max(x: f64, y: f64) -> f64 {
        libm::fmax(x, y)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn exp(x: f64) -> f64 {
        x.exp()
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn exp(x: f64) -> f64 {
        libm::exp(x)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn ln(x: f64) -> f64 {
        x.ln()
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn ln(x: f64) -> f64 {
        libm::log(x)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn sin(x: f64) -> f64 {
        x.sin()
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn sin(x: f64) -> f64 {
        libm::sin(x)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn cos(x: f64) -> f64 {
        x.cos()
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn cos(x: f64) -> f64 {
        libm::cos(x)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn acos(x: f64) -> f64 {
        x.acos()
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn acos(x: f64) -> f64 {
        libm::acos(x)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn asin(x: f64) -> f64 {
        x.asin()
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn asin(x: f64) -> f64 {
        libm::asin(x)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn atan2(y: f64, x: f64) -> f64 {
        y.atan2(x)
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn atan2(y: f64, x: f64) -> f64 {
        libm::atan2(y, x)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn powi(x: f64, n: i32) -> f64 {
        x.powi(n)
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn powi(x: f64, n: i32) -> f64 {
        libm::pow(x, n as f64)
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn powf(x: f64, y: f64) -> f64 {
        x.powf(y)
    }

    #[cfg(all(feature = "libm", not(feature = "std")))]
    #[inline(always)]
    pub fn powf(x: f64, y: f64) -> f64 {
        libm::pow(x, y)
    }
}

// ============================================================================
// Scalar Mask Types
// ============================================================================

/// Boolean mask wrapper for `f32` scalar operations.
///
/// This newtype wrapper provides a consistent API for mask operations
/// across both `f32` and `f64` types, enabling symmetric code patterns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskF32(pub bool);

/// Boolean mask wrapper for `f64` scalar operations.
///
/// This newtype wrapper provides a consistent API for mask operations
/// across both `f32` and `f64` types, enabling symmetric code patterns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskF64(pub bool);

// ============================================================================
// MaskF32 Implementation
// ============================================================================

impl BitAnd for MaskF32 {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 && rhs.0)
    }
}

impl BitOr for MaskF32 {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 || rhs.0)
    }
}

impl Not for MaskF32 {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

impl Mask for MaskF32 {
    type Vector = f32;

    #[inline(always)]
    fn any(self) -> bool {
        self.0
    }

    #[inline(always)]
    fn all(self) -> bool {
        self.0
    }

    #[inline(always)]
    fn select(self, if_true: f32, if_false: f32) -> f32 {
        if self.0 { if_true } else { if_false }
    }
}

// ============================================================================
// MaskF64 Implementation
// ============================================================================

impl BitAnd for MaskF64 {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 && rhs.0)
    }
}

impl BitOr for MaskF64 {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 || rhs.0)
    }
}

impl Not for MaskF64 {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

impl Mask for MaskF64 {
    type Vector = f64;

    #[inline(always)]
    fn any(self) -> bool {
        self.0
    }

    #[inline(always)]
    fn all(self) -> bool {
        self.0
    }

    #[inline(always)]
    fn select(self, if_true: f64, if_false: f64) -> f64 {
        if self.0 { if_true } else { if_false }
    }
}

// ============================================================================
// Scalar Implementation: f32 as Vector
// ============================================================================

impl Vector for f32 {
    type Mask = MaskF32;
    type Scalar = f32;
    const LANES: usize = 1;

    #[inline(always)]
    fn splat(value: f64) -> Self {
        value as f32
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        scalar_f32::sqrt(self)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        scalar_f32::abs(self)
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        scalar_f32::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        scalar_f32::max(self, other)
    }

    #[inline(always)]
    fn exp(self) -> Self {
        scalar_f32::exp(self)
    }

    #[inline(always)]
    fn ln(self) -> Self {
        scalar_f32::ln(self)
    }

    #[inline(always)]
    fn sin(self) -> Self {
        scalar_f32::sin(self)
    }

    #[inline(always)]
    fn cos(self) -> Self {
        scalar_f32::cos(self)
    }

    #[inline(always)]
    fn acos(self) -> Self {
        scalar_f32::acos(self)
    }

    #[inline(always)]
    fn asin(self) -> Self {
        scalar_f32::asin(self)
    }

    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        scalar_f32::atan2(self, other)
    }

    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        scalar_f32::powi(self, n)
    }

    #[inline(always)]
    fn powf(self, exp: Self) -> Self {
        scalar_f32::powf(self, exp)
    }

    #[inline(always)]
    fn lt(self, other: Self) -> Self::Mask {
        MaskF32(self < other)
    }

    #[inline(always)]
    fn le(self, other: Self) -> Self::Mask {
        MaskF32(self <= other)
    }

    #[inline(always)]
    fn gt(self, other: Self) -> Self::Mask {
        MaskF32(self > other)
    }

    #[inline(always)]
    fn ge(self, other: Self) -> Self::Mask {
        MaskF32(self >= other)
    }

    #[inline(always)]
    fn eq(self, other: Self) -> Self::Mask {
        MaskF32(self == other)
    }
}

// ============================================================================
// Scalar Implementation: f64 as Vector
// ============================================================================

impl Vector for f64 {
    type Mask = MaskF64;
    type Scalar = f64;
    const LANES: usize = 1;

    #[inline(always)]
    fn splat(value: f64) -> Self {
        value
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        scalar_f64::sqrt(self)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        scalar_f64::abs(self)
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        scalar_f64::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        scalar_f64::max(self, other)
    }

    #[inline(always)]
    fn exp(self) -> Self {
        scalar_f64::exp(self)
    }

    #[inline(always)]
    fn ln(self) -> Self {
        scalar_f64::ln(self)
    }

    #[inline(always)]
    fn sin(self) -> Self {
        scalar_f64::sin(self)
    }

    #[inline(always)]
    fn cos(self) -> Self {
        scalar_f64::cos(self)
    }

    #[inline(always)]
    fn acos(self) -> Self {
        scalar_f64::acos(self)
    }

    #[inline(always)]
    fn asin(self) -> Self {
        scalar_f64::asin(self)
    }

    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        scalar_f64::atan2(self, other)
    }

    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        scalar_f64::powi(self, n)
    }

    #[inline(always)]
    fn powf(self, exp: Self) -> Self {
        scalar_f64::powf(self, exp)
    }

    #[inline(always)]
    fn lt(self, other: Self) -> Self::Mask {
        MaskF64(self < other)
    }

    #[inline(always)]
    fn le(self, other: Self) -> Self::Mask {
        MaskF64(self <= other)
    }

    #[inline(always)]
    fn gt(self, other: Self) -> Self::Mask {
        MaskF64(self > other)
    }

    #[inline(always)]
    fn ge(self, other: Self) -> Self::Mask {
        MaskF64(self >= other)
    }

    #[inline(always)]
    fn eq(self, other: Self) -> Self::Mask {
        MaskF64(self == other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_f64(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps || (a - b).abs() / a.abs().max(b.abs()).max(1.0) < eps
    }

    fn approx_eq_f32(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps || (a - b).abs() / a.abs().max(b.abs()).max(1.0) < eps
    }

    // ========================================================================
    // f32 Tests
    // ========================================================================

    #[test]
    fn test_f32_splat() {
        let val = 42.5;
        assert!(approx_eq_f32(f32::splat(val), val as f32, 1e-5));
    }

    #[test]
    fn test_f32_zero_and_one() {
        assert_eq!(f32::zero(), 0.0_f32);
        assert_eq!(f32::one(), 1.0_f32);
    }

    #[test]
    fn test_f32_sqrt() {
        assert!(approx_eq_f32(4.0_f32.sqrt(), 2.0, 1e-6));
        assert!(approx_eq_f32(9.0_f32.sqrt(), 3.0, 1e-6));
    }

    #[test]
    fn test_f32_rsqrt() {
        assert!(approx_eq_f32(4.0_f32.rsqrt(), 0.5, 1e-6));
    }

    #[test]
    fn test_f32_recip() {
        assert!(approx_eq_f32(2.0_f32.recip(), 0.5, 1e-6));
    }

    #[test]
    fn test_f32_abs() {
        assert!(approx_eq_f32((-3.0_f32).abs(), 3.0, 1e-6));
        assert!(approx_eq_f32(3.0_f32.abs(), 3.0, 1e-6));
    }

    #[test]
    fn test_f32_min_max() {
        assert_eq!(3.0_f32.min(5.0), 3.0);
        assert_eq!(3.0_f32.max(5.0), 5.0);
    }

    #[test]
    fn test_f32_exp_ln() {
        assert!(approx_eq_f32(1.0_f32.exp(), core::f32::consts::E, 1e-6));
        assert!(approx_eq_f32(core::f32::consts::E.ln(), 1.0, 1e-6));
    }

    #[test]
    fn test_f32_sin_cos() {
        assert!(approx_eq_f32(0.0_f32.sin(), 0.0, 1e-6));
        assert!(approx_eq_f32(0.0_f32.cos(), 1.0, 1e-6));
    }

    #[test]
    fn test_f32_acos_asin() {
        assert!(approx_eq_f32(
            0.0_f32.acos(),
            core::f32::consts::FRAC_PI_2,
            1e-6
        ));
        assert!(approx_eq_f32(0.0_f32.asin(), 0.0, 1e-6));
    }

    #[test]
    fn test_f32_atan2() {
        assert!(approx_eq_f32(
            1.0_f32.atan2(1.0),
            core::f32::consts::FRAC_PI_4,
            1e-6
        ));
    }

    #[test]
    fn test_f32_powi_powf() {
        assert!(approx_eq_f32(2.0_f32.powi(3), 8.0, 1e-6));
        assert!(approx_eq_f32(2.0_f32.powf(3.0), 8.0, 1e-6));
    }

    #[test]
    fn test_f32_comparisons() {
        assert!(1.0_f32.lt(2.0).0);
        assert!(!2.0_f32.lt(1.0).0);
        assert!(1.0_f32.le(1.0).0);
        assert!(2.0_f32.gt(1.0).0);
        assert!(2.0_f32.ge(2.0).0);
        assert!(1.0_f32.eq(1.0).0);
    }

    #[test]
    fn test_f32_mask_select() {
        assert_eq!(MaskF32(true).select(1.0_f32, 2.0_f32), 1.0_f32);
        assert_eq!(MaskF32(false).select(1.0_f32, 2.0_f32), 2.0_f32);
    }

    #[test]
    fn test_f32_mask_any_all() {
        assert!(MaskF32(true).any());
        assert!(MaskF32(true).all());
        assert!(!MaskF32(false).any());
        assert!(!MaskF32(false).all());
    }

    #[test]
    fn test_f32_mask_bitops() {
        assert_eq!(MaskF32(true) & MaskF32(true), MaskF32(true));
        assert_eq!(MaskF32(true) & MaskF32(false), MaskF32(false));
        assert_eq!(MaskF32(true) | MaskF32(false), MaskF32(true));
        assert_eq!(MaskF32(false) | MaskF32(false), MaskF32(false));
        assert_eq!(!MaskF32(true), MaskF32(false));
        assert_eq!(!MaskF32(false), MaskF32(true));
    }

    // ========================================================================
    // f64 Tests
    // ========================================================================

    #[test]
    fn test_f64_splat() {
        let val = 42.5;
        assert_eq!(f64::splat(val), val);
    }

    #[test]
    fn test_f64_zero_and_one() {
        assert_eq!(f64::zero(), 0.0);
        assert_eq!(f64::one(), 1.0);
    }

    #[test]
    fn test_f64_sqrt() {
        assert!(approx_eq_f64(4.0_f64.sqrt(), 2.0, 1e-10));
        assert!(approx_eq_f64(9.0_f64.sqrt(), 3.0, 1e-10));
    }

    #[test]
    fn test_f64_rsqrt() {
        assert!(approx_eq_f64(4.0_f64.rsqrt(), 0.5, 1e-10));
    }

    #[test]
    fn test_f64_recip() {
        assert!(approx_eq_f64(2.0_f64.recip(), 0.5, 1e-10));
    }

    #[test]
    fn test_f64_abs() {
        assert!(approx_eq_f64((-3.0_f64).abs(), 3.0, 1e-10));
        assert!(approx_eq_f64(3.0_f64.abs(), 3.0, 1e-10));
    }

    #[test]
    fn test_f64_min_max() {
        assert_eq!(3.0_f64.min(5.0), 3.0);
        assert_eq!(3.0_f64.max(5.0), 5.0);
    }

    #[test]
    fn test_f64_exp_ln() {
        assert!(approx_eq_f64(1.0_f64.exp(), core::f64::consts::E, 1e-10));
        assert!(approx_eq_f64(core::f64::consts::E.ln(), 1.0, 1e-10));
    }

    #[test]
    fn test_f64_sin_cos() {
        assert!(approx_eq_f64(0.0_f64.sin(), 0.0, 1e-10));
        assert!(approx_eq_f64(0.0_f64.cos(), 1.0, 1e-10));
    }

    #[test]
    fn test_f64_acos_asin() {
        assert!(approx_eq_f64(
            0.0_f64.acos(),
            core::f64::consts::FRAC_PI_2,
            1e-10
        ));
        assert!(approx_eq_f64(0.0_f64.asin(), 0.0, 1e-10));
    }

    #[test]
    fn test_f64_atan2() {
        assert!(approx_eq_f64(
            1.0_f64.atan2(1.0),
            core::f64::consts::FRAC_PI_4,
            1e-10
        ));
    }

    #[test]
    fn test_f64_powi_powf() {
        assert!(approx_eq_f64(2.0_f64.powi(3), 8.0, 1e-10));
        assert!(approx_eq_f64(2.0_f64.powf(3.0), 8.0, 1e-10));
    }

    #[test]
    fn test_f64_comparisons() {
        assert!(1.0_f64.lt(2.0).0);
        assert!(!2.0_f64.lt(1.0).0);
        assert!(1.0_f64.le(1.0).0);
        assert!(2.0_f64.gt(1.0).0);
        assert!(2.0_f64.ge(2.0).0);
        assert!(1.0_f64.eq(1.0).0);
    }

    #[test]
    fn test_f64_mask_select() {
        assert_eq!(MaskF64(true).select(1.0, 2.0), 1.0);
        assert_eq!(MaskF64(false).select(1.0, 2.0), 2.0);
    }

    #[test]
    fn test_f64_mask_any_all() {
        assert!(MaskF64(true).any());
        assert!(MaskF64(true).all());
        assert!(!MaskF64(false).any());
        assert!(!MaskF64(false).all());
    }

    #[test]
    fn test_f64_mask_bitops() {
        assert_eq!(MaskF64(true) & MaskF64(true), MaskF64(true));
        assert_eq!(MaskF64(true) & MaskF64(false), MaskF64(false));
        assert_eq!(MaskF64(true) | MaskF64(false), MaskF64(true));
        assert_eq!(MaskF64(false) | MaskF64(false), MaskF64(false));
        assert_eq!(!MaskF64(true), MaskF64(false));
        assert_eq!(!MaskF64(false), MaskF64(true));
    }

    // ========================================================================
    // Float Trait Tests
    // ========================================================================

    #[test]
    fn test_float_constants() {
        assert!(approx_eq_f32(f32::PI, core::f32::consts::PI, 1e-7));
        assert!(approx_eq_f64(f64::PI, core::f64::consts::PI, 1e-15));
    }

    #[test]
    fn test_float_conversions() {
        assert_eq!(f64::from_f64(42.0), 42.0);
        assert!(approx_eq_f32(f32::from_f64(42.0), 42.0, 1e-6));
        assert_eq!(42.0_f64.to_f64(), 42.0);
        assert!(approx_eq_f64(42.0_f32.to_f64(), 42.0, 1e-6));
    }

    // ========================================================================
    // Vector Trait Associated Constants Tests
    // ========================================================================

    #[test]
    fn test_vector_lanes() {
        assert_eq!(f64::LANES, 1);
        assert_eq!(f32::LANES, 1);
    }
}
