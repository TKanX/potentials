//! # Meta Potentials (Wrappers and Modifiers)
//!
//! Generic wrappers that modify the behavior of base potentials.
//!
//! ## Available Types
//!
//! | Type | Description | Formula |
//! |------|-------------|---------|
//! | [`Cutoff`] | Hard cutoff | `V(r) if r < rc, else 0` |
//! | [`Shift`] | Shifted potential | `V(r) - V(rc)` |
//! | [`Switch`] | Smooth switching | `V(r) * S(r)` |
//! | [`Softcore`] | Softcore for FEP | Modified for alchemical |
//! | [`Scaled`] | Linear scaling | `lambda * V(r)` |
//! | [`Sum`] | Potential sum | `V1(r) + V2(r)` |

mod cutoff;
mod scaled;
mod shift;
mod softcore;
mod sum;
mod switch;

pub use cutoff::Cutoff;
pub use scaled::Scaled;
pub use shift::Shift;
pub use softcore::Softcore;
pub use sum::Sum;
pub use switch::Switch;
