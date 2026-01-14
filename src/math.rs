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
//! | Type | Description |
//! |------|-------------|
//! | `f32` | 32-bit floating point |
//! | `f64` | 64-bit floating point |

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
