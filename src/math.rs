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
