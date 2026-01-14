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
