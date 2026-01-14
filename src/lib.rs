//! # Potentials
//!
//! **A high-performance, pure Rust library for classical molecular dynamics potential energy functions.**
//!
//! *Designed for force field engines, MD simulators, and scientific computing in embedded environments.*
//!
//! [Features](#features) • [Installation](#installation) • [Usage](#usage) • [Modules](#modules) • [Performance](#performance) • [License](#license)
//!
//! ---
//!
//! ## Features
//!
//! - **🚀 High Performance**
//!
//!   - **Zero Heap Allocation**: All computations happen on the stack
//!   - **Branchless Kernels**: Mask-based conditionals for vectorization-friendly code
//!   - **Optimized Implementations**: Shared subexpressions, Horner's method, Chebyshev recursion
//!   - **~650M interactions/sec**: LJ energy+force throughput on modern CPUs
//!
//! - **🎯 Comprehensive Coverage**
//!
//!   - **7 Pair Potentials**: LJ, Mie, Buckingham, Coulomb, Yukawa, Gaussian, Soft sphere
//!   - **6 Bond Potentials**: Harmonic, GROMOS96, Morse, FENE, Cubic, Quartic
//!   - **5 Angle Potentials**: Harmonic, Cosine, Linear, Urey-Bradley, Cross
//!   - **4 Torsion Potentials**: Periodic cosine, OPLS, Ryckaert-Bellemans, Harmonic
//!   - **3 Improper Potentials**: Harmonic, Cosine, Distance-based
//!   - **2 H-Bond Potentials**: DREIDING 12-10, LJ-Cosine
//!   - **6 Meta Wrappers**: Cutoff, Shift, Switch, Softcore, Scaled, Sum
//!
//! - **🔬 Physically Accurate**
//!
//!   - Analytically correct energy and force expressions
//!   - Numerical derivative validation for every potential
//!   - Proper handling of edge cases and singularities
//!   - CODATA 2018 physical constants
//!
//! - **🛰️ Embeddable & Portable**
//!
//!   - **`no_std` Support**: Works in embedded devices, kernels, or WASM
//!   - **Pure Rust**: No C/C++ dependencies, simple build chain
//!   - **Generic Precision**: Supports both `f32` and `f64`
//!
//! ## Installation
//!
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! potentials = "0.1.0"
//! ```
//!
//! To use in a `no_std` environment:
//!
//! ```toml
//! [dependencies]
//! potentials = { version = "0.1.0", default-features = false, features = ["libm"] }
//! ```
//!
//! ## Usage
//!
//! ### Quick Start
//!
//! ```
//! use potentials::{pair::Lj, base::Potential2};
//!
//! // Create Lennard-Jones potential (ε=1.0 kcal/mol, σ=3.4 Å)
//! let lj = Lj::<f64>::new(1.0, 3.4);
//!
//! // Compute energy and force factor at r=4.0 Å
//! let r_sq = 16.0;  // r² = 4²
//! let (energy, force_factor) = lj.energy_force(r_sq);
//!
//! // Force vector: F_ij = force_factor * r_ij_vec
//! println!("Energy: {:.6} kcal/mol", energy);
//! println!("Force factor: {:.6}", force_factor);
//! ```
//!
//! ### Adding Cutoffs and Modifications
//!
//! ```
//! use potentials::{pair::Lj, meta::Switch, base::Potential2};
//!
//! let lj = Lj::<f64>::new(1.0, 3.4);
//!
//! // Smooth switching function
//! let lj_switched: Switch<_, f64> = Switch::new(lj, 9.0, 12.0);
//!
//! // Energy smoothly goes to zero between r=9 and r=12 Å
//! let energy = lj_switched.energy(100.0);  // r² = 100
//! ```
//!
//! ### Combining Potentials
//!
//! ```
//! use potentials::{pair::{Lj, Coul}, meta::Sum, base::Potential2};
//!
//! let lj = Lj::<f64>::new(0.1, 3.0);
//! let coul = Coul::<f64>::new(-332.0636);  // Na⁺-Cl⁻ attraction
//!
//! // LJ + Coulomb combined potential
//! let combined: Sum<_, _, f64> = Sum::new(lj, coul);
//!
//! let (energy, force_factor) = combined.energy_force(25.0);  // r² = 25
//! ```
//!
//! ### Angle and Torsion Potentials
//!
//! ```
//! use potentials::{angle::Cos, torsion::Opls, base::{Potential3, Potential4}};
//!
//! // Water H-O-H angle (cosine form, faster than harmonic)
//! let angle = Cos::<f64>::from_degrees(100.0, 104.5);
//! let (e_angle, d_angle) = angle.energy_derivative(1.0, 1.0, 0.25);
//!
//! // Alkane torsion (OPLS Fourier series)
//! let torsion = Opls::<f64>::new(0.7, 0.1, 1.5, 0.0);
//! let (e_tors, d_tors) = torsion.energy_derivative(0.5, 0.866);
//! ```
//!
//! ## Modules
//!
//! ### Architecture
//!
//! The library is organized around three core traits:
//!
//! | Trait | Bodies | Input | Output | Use Case |
//! | ----- | ------ | ----- | ------ | -------- |
//! | [`Potential2`](base::Potential2) | 2 | `r²` | `(V, S)` | Pairs, Bonds |
//! | [`Potential3`](base::Potential3) | 3 | `r_ij², r_jk², cos θ` | `(V, dV/d(cos θ))` | Angles |
//! | [`Potential4`](base::Potential4) | 4 | `cos φ, sin φ` | `(V, dV/dφ)` | Torsions |
//!
//! ### Module Reference
//!
//! | Module | Potentials | Description |
//! | ------ | ---------- | ----------- |
//! | [`pair`] | [`Lj`](pair::Lj), [`Mie`](pair::Mie), [`Buck`](pair::Buck), [`Coul`](pair::Coul), [`Yukawa`](pair::Yukawa), [`Gauss`](pair::Gauss), [`Soft`](pair::Soft) | Non-bonded pair interactions |
//! | [`bond`] | [`Harm`](bond::Harm), [`G96`](bond::G96), [`Morse`](bond::Morse), [`Fene`](bond::Fene), [`Cubic`](bond::Cubic), [`Quart`](bond::Quart) | Covalent bond stretching |
//! | [`angle`] | [`Harm`](angle::Harm), [`Cos`](angle::Cos), [`Linear`](angle::Linear), [`Urey`](angle::Urey), [`Cross`](angle::Cross) | Valence angle bending |
//! | [`torsion`] | [`Cos`](torsion::Cos), [`Opls`](torsion::Opls), [`Rb`](torsion::Rb), [`Harm`](torsion::Harm) | Proper dihedral angles |
//! | [`imp`] | [`Harm`](imp::Harm), [`Cos`](imp::Cos), [`Dist`](imp::Dist) | Improper torsions |
//! | [`hbond`] | [`Dreid`](hbond::Dreid), [`LjCos`](hbond::LjCos) | Hydrogen bonding |
//! | [`meta`] | [`Cutoff`](meta::Cutoff), [`Shift`](meta::Shift), [`Switch`](meta::Switch), [`Softcore`](meta::Softcore), [`Scaled`](meta::Scaled), [`Sum`](meta::Sum) | Potential modifiers |
//! | [`consts`] | — | Physical constants (CODATA 2018) |
//!
//! ### Force Field Compatibility
//!
//! | Force Field | Pair | Bond | Angle | Torsion | Improper |
//! | ----------- | :--: | :--: | :---: | :-----: | :------: |
//! | AMBER       |  ✅  |  ✅  |  ✅   |   ✅    |    ✅    |
//! | CHARMM      |  ✅  |  ✅  |  ✅   |   ✅    |    ✅    |
//! | OPLS        |  ✅  |  ✅  |  ✅   |   ✅    |    ✅    |
//! | GROMOS      |  ✅  |  ✅  |  ✅   |   ✅    |    ✅    |
//! | DREIDING    |  ✅  |  ✅  |  ✅   |   ✅    |    ✅    |
//!
//! ## Performance
//!
//! Benchmarked on an Intel Core i7-13620H Laptop CPU (10 cores).
//!
//! ### Pair Potentials (energy + force)
//!
//! | Potential | Throughput | Time per eval |
//! | --------- | ---------- | ------------- |
//! | **Lennard-Jones** | **200 M/s** | **5.0 ns** |
//! | Coulomb | 69 M/s | 14.5 ns |
//! | Gaussian | 57 M/s | 17.6 ns |
//! | Soft sphere | 53 M/s | 19.0 ns |
//! | Yukawa | 41 M/s | 24.4 ns |
//! | Buckingham | 36 M/s | 27.4 ns |
//! | Mie (n-m) | 28 M/s | 36.3 ns |
//!
//! ### Angle Potentials
//!
//! | Potential | Throughput | Notes |
//! | --------- | ---------- | ----- |
//! | **Cosine** | **367 M/s** | No acos needed |
//! | Harmonic | 54 M/s | Requires acos |
//!
//! ### Meta Wrapper Overhead
//!
//! | Configuration | Throughput | Overhead |
//! | ------------- | ---------- | -------- |
//! | LJ bare | 200 M/s | baseline |
//! | LJ + softcore | 113 M/s | 1.8× |
//! | LJ + cutoff | 107 M/s | 1.9× |
//! | LJ + shift | 100 M/s | 2.0× |
//! | LJ + switch | 76 M/s | 2.6× |
//!
//! ### Throughput Scaling
//!
//! | Interactions | Throughput | Notes |
//! | ------------ | ---------- | ----- |
//! | 1,000 | 660 M/s | Cache-hot |
//! | 10,000 | 648 M/s | L2 cache |
//! | 100,000 | 649 M/s | L3 cache |
//!
//! ## License
//!
//! This project is licensed under the MIT License - see the [LICENSE](https://github.com/TKanX/potentials/blob/main/LICENSE) file for details.
//!
//! ---
//!
//! **Made with ❤️ for the scientific computing community**

#![no_std]

// Core modules
pub mod base;
pub mod consts;
pub mod math;

// Potential types
pub mod angle;
pub mod bond;
pub mod hbond;
pub mod imp;
pub mod pair;
pub mod torsion;

// Meta wrappers
pub mod meta;
