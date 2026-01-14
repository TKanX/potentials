//! # Physical Constants
//!
//! Fundamental physical constants used in molecular dynamics simulations.
//! All values are in SI units unless otherwise noted.
//!
//! ## Unit Systems
//!
//! This library uses "natural" MD units internally. Constants are provided
//! for unit conversion when interfacing with external data.
//!
//! Common MD unit systems:
//!
//! | System    | Length | Energy    | Time  | Mass    |
//! |-----------|--------|-----------|-------|---------|
//! | Real      | A      | kcal/mol  | fs    | g/mol   |
//! | Metal     | A      | eV        | ps    | g/mol   |
//! | SI        | m      | J         | s     | kg      |
//! | LJ        | sigma  | epsilon   | tau   | m       |

// ============================================================================
// Mathematical Constants
// ============================================================================

/// Pi (π)
pub const PI: f64 = core::f64::consts::PI;

/// 2 * Pi
pub const TWO_PI: f64 = 2.0 * PI;

/// Pi / 2
pub const HALF_PI: f64 = PI / 2.0;

/// Pi / 180 (degrees to radians)
pub const DEG_TO_RAD: f64 = PI / 180.0;

/// 180 / Pi (radians to degrees)
pub const RAD_TO_DEG: f64 = 180.0 / PI;

// ============================================================================
// Fundamental Physical Constants (CODATA 2018)
// ============================================================================

/// Speed of light in vacuum (m/s)
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Planck constant (J·s)
pub const PLANCK: f64 = 6.626_070_15e-34;

/// Reduced Planck constant (h-bar) (J·s)
pub const HBAR: f64 = 1.054_571_817e-34;

/// Elementary charge (C)
pub const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;

/// Electron mass (kg)
pub const ELECTRON_MASS: f64 = 9.109_383_701_5e-31;

/// Proton mass (kg)
pub const PROTON_MASS: f64 = 1.672_621_923_69e-27;

/// Avogadro constant (1/mol)
pub const AVOGADRO: f64 = 6.022_140_76e23;

/// Boltzmann constant (J/K)
pub const BOLTZMANN: f64 = 1.380_649e-23;

/// Molar gas constant R = N_A * k_B (J/(mol·K))
pub const GAS_CONSTANT: f64 = 8.314_462_618;

/// Vacuum permittivity (epsilon_0) (F/m)
pub const VACUUM_PERMITTIVITY: f64 = 8.854_187_812_8e-12;

/// Coulomb constant: 1/(4*pi*epsilon_0) (N·m²/C²)
pub const COULOMB_CONSTANT: f64 = 8.987_551_792_3e9;

// ============================================================================
// Derived Constants for MD
// ============================================================================

/// Atomic mass unit (kg)
pub const AMU: f64 = 1.660_539_066_60e-27;

/// Angstrom in meters (m)
pub const ANGSTROM: f64 = 1.0e-10;

/// Femtosecond in seconds (s)
pub const FEMTOSECOND: f64 = 1.0e-15;

/// Picosecond in seconds (s)
pub const PICOSECOND: f64 = 1.0e-12;

/// Electron volt in Joules (J)
pub const ELECTRONVOLT: f64 = 1.602_176_634e-19;

/// kcal/mol in Joules (J/mol)
pub const KCAL_PER_MOL: f64 = 4184.0;

/// kJ/mol in Joules (J/mol)
pub const KJ_PER_MOL: f64 = 1000.0;

// ============================================================================
// Electrostatics Conversion Factors
// ============================================================================

/// Coulomb prefactor for "real" units: e^2/(4*pi*eps_0) in [kcal·A/mol·e^2]
///
/// Used to convert: V = COULOMB_REAL * q1 * q2 / r
/// where q in elementary charges, r in Angstroms, V in kcal/mol
pub const COULOMB_REAL: f64 = 332.063_713;

/// Coulomb prefactor for "metal" units: e^2/(4*pi*eps_0) in [eV·A/e^2]
///
/// Used when energy is in eV, distance in Angstroms
pub const COULOMB_METAL: f64 = 14.399_645;

// ============================================================================
// Unit Conversion Functions
// ============================================================================

/// Converts angle from degrees to radians.
#[inline]
pub const fn deg2rad(deg: f64) -> f64 {
    deg * DEG_TO_RAD
}

/// Converts angle from radians to degrees.
#[inline]
pub const fn rad2deg(rad: f64) -> f64 {
    rad * RAD_TO_DEG
}

/// Converts energy from kcal/mol to kJ/mol.
#[inline]
pub const fn kcal_to_kj(kcal: f64) -> f64 {
    kcal * 4.184
}

/// Converts energy from kJ/mol to kcal/mol.
#[inline]
pub const fn kj_to_kcal(kj: f64) -> f64 {
    kj / 4.184
}

/// Converts energy from eV to kcal/mol.
#[inline]
pub const fn ev_to_kcal(ev: f64) -> f64 {
    ev * 23.060_541
}

/// Converts energy from kcal/mol to eV.
#[inline]
pub const fn kcal_to_ev(kcal: f64) -> f64 {
    kcal / 23.060_541
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ========================================================================
    // Mathematical Constants Tests
    // ========================================================================

    #[test]
    fn test_pi() {
        assert_relative_eq!(PI, core::f64::consts::PI, epsilon = 1e-15);
    }

    #[test]
    fn test_two_pi() {
        assert_relative_eq!(TWO_PI, 2.0 * PI, epsilon = 1e-15);
    }

    #[test]
    fn test_half_pi() {
        assert_relative_eq!(HALF_PI, PI / 2.0, epsilon = 1e-15);
    }

    #[test]
    fn test_deg_to_rad() {
        assert_relative_eq!(180.0 * DEG_TO_RAD, PI, epsilon = 1e-15);
    }

    #[test]
    fn test_rad_to_deg() {
        assert_relative_eq!(PI * RAD_TO_DEG, 180.0, epsilon = 1e-15);
    }

    #[test]
    fn test_deg_rad_inverse() {
        assert_relative_eq!(DEG_TO_RAD * RAD_TO_DEG, 1.0, epsilon = 1e-15);
    }

    // ========================================================================
    // Fundamental Physical Constants Tests (CODATA 2018)
    // ========================================================================

    #[test]
    fn test_speed_of_light() {
        assert_eq!(SPEED_OF_LIGHT, 299_792_458.0);
    }

    #[test]
    fn test_planck_constant() {
        assert_eq!(PLANCK, 6.626_070_15e-34);
    }

    #[test]
    fn test_hbar() {
        assert_relative_eq!(HBAR, PLANCK / TWO_PI, epsilon = 1e-20);
    }

    #[test]
    fn test_elementary_charge() {
        assert_eq!(ELEMENTARY_CHARGE, 1.602_176_634e-19);
    }

    #[test]
    fn test_electron_mass() {
        assert_relative_eq!(ELECTRON_MASS, 9.109_383_701_5e-31, epsilon = 1e-40);
    }

    #[test]
    fn test_proton_mass() {
        assert_relative_eq!(PROTON_MASS, 1.672_621_923_69e-27, epsilon = 1e-37);
    }

    #[test]
    fn test_avogadro() {
        assert_eq!(AVOGADRO, 6.022_140_76e23);
    }

    #[test]
    fn test_boltzmann() {
        assert_eq!(BOLTZMANN, 1.380_649e-23);
    }

    #[test]
    fn test_gas_constant() {
        assert_relative_eq!(GAS_CONSTANT, AVOGADRO * BOLTZMANN, epsilon = 1e-6);
    }

    #[test]
    fn test_vacuum_permittivity() {
        assert_relative_eq!(VACUUM_PERMITTIVITY, 8.854_187_812_8e-12, epsilon = 1e-21);
    }

    #[test]
    fn test_coulomb_constant() {
        let calculated = 1.0 / (4.0 * PI * VACUUM_PERMITTIVITY);
        assert_relative_eq!(COULOMB_CONSTANT, calculated, max_relative = 1e-8);
    }

    // ========================================================================
    // Derived Constants Tests
    // ========================================================================

    #[test]
    fn test_amu() {
        assert_relative_eq!(AMU, 1.660_539_066_60e-27, epsilon = 1e-36);
    }

    #[test]
    fn test_angstrom() {
        assert_eq!(ANGSTROM, 1.0e-10);
    }

    #[test]
    fn test_femtosecond() {
        assert_eq!(FEMTOSECOND, 1.0e-15);
    }

    #[test]
    fn test_picosecond() {
        assert_eq!(PICOSECOND, 1.0e-12);
        assert_eq!(PICOSECOND, 1000.0 * FEMTOSECOND);
    }

    #[test]
    fn test_electronvolt() {
        assert_eq!(ELECTRONVOLT, ELEMENTARY_CHARGE);
    }

    #[test]
    fn test_kcal_per_mol() {
        assert_eq!(KCAL_PER_MOL, 4184.0);
    }

    #[test]
    fn test_kj_per_mol() {
        assert_eq!(KJ_PER_MOL, 1000.0);
    }

    // ========================================================================
    // Electrostatics Conversion Factors Tests
    // ========================================================================

    #[test]
    fn test_coulomb_real() {
        assert_relative_eq!(COULOMB_REAL, 332.063_713, epsilon = 1e-6);
    }

    #[test]
    fn test_coulomb_metal() {
        assert_relative_eq!(COULOMB_METAL, 14.399_645, epsilon = 1e-6);
    }

    #[test]
    fn test_coulomb_real_metal_ratio() {
        let ratio = COULOMB_REAL / COULOMB_METAL;
        assert_relative_eq!(ratio, 23.060_541, epsilon = 1e-4);
    }

    // ========================================================================
    // Conversion Function Tests
    // ========================================================================

    #[test]
    fn test_deg2rad() {
        assert_relative_eq!(deg2rad(0.0), 0.0, epsilon = 1e-15);
        assert_relative_eq!(deg2rad(90.0), HALF_PI, epsilon = 1e-15);
        assert_relative_eq!(deg2rad(180.0), PI, epsilon = 1e-15);
        assert_relative_eq!(deg2rad(360.0), TWO_PI, epsilon = 1e-15);
    }

    #[test]
    fn test_rad2deg() {
        assert_relative_eq!(rad2deg(0.0), 0.0, epsilon = 1e-15);
        assert_relative_eq!(rad2deg(HALF_PI), 90.0, epsilon = 1e-10);
        assert_relative_eq!(rad2deg(PI), 180.0, epsilon = 1e-10);
        assert_relative_eq!(rad2deg(TWO_PI), 360.0, epsilon = 1e-10);
    }

    #[test]
    fn test_deg_rad_roundtrip() {
        let values = [0.0, 45.0, 90.0, 180.0, 270.0, 360.0];
        for &deg in &values {
            assert_relative_eq!(rad2deg(deg2rad(deg)), deg, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_kcal_to_kj() {
        assert_relative_eq!(kcal_to_kj(1.0), 4.184, epsilon = 1e-10);
        assert_relative_eq!(kcal_to_kj(0.0), 0.0, epsilon = 1e-15);
        assert_relative_eq!(kcal_to_kj(10.0), 41.84, epsilon = 1e-10);
    }

    #[test]
    fn test_kj_to_kcal() {
        assert_relative_eq!(kj_to_kcal(4.184), 1.0, epsilon = 1e-10);
        assert_relative_eq!(kj_to_kcal(0.0), 0.0, epsilon = 1e-15);
        assert_relative_eq!(kj_to_kcal(41.84), 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kcal_kj_roundtrip() {
        let values = [0.0, 1.0, 10.0, 100.0, -5.0];
        for &kcal in &values {
            assert_relative_eq!(kj_to_kcal(kcal_to_kj(kcal)), kcal, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_ev_to_kcal() {
        assert_relative_eq!(ev_to_kcal(1.0), 23.060_541, epsilon = 1e-6);
        assert_relative_eq!(ev_to_kcal(0.0), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_kcal_to_ev() {
        assert_relative_eq!(kcal_to_ev(23.060_541), 1.0, epsilon = 1e-6);
        assert_relative_eq!(kcal_to_ev(0.0), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_ev_kcal_roundtrip() {
        let values = [0.0, 1.0, 10.0, 100.0, -5.0];
        for &ev in &values {
            assert_relative_eq!(kcal_to_ev(ev_to_kcal(ev)), ev, epsilon = 1e-10);
        }
    }
}
