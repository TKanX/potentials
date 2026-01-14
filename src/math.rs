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
