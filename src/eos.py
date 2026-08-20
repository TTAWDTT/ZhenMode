"""
Equation of State for seawater.

Two options:
  1. Linearized Boussinesq (default, fast):
     rho = rho_0 * [1 - alpha*(T-T_ref) + beta*(S-S_ref)]

  2. UNESCO polynomial (nonlinear, more accurate):
     Full 1980 UNESCO International EOS, valid for T in [-2, 40] C,
     S in [0, 42] psu, P at atmospheric (0 Pa gauge).

For the Boussinesq approximation, density only enters through the
buoyancy term in the momentum equation and the hydrostatic pressure
computation. The inertia term uses rho_0 everywhere.
"""
import numpy as np

from config import RHO_0, ALPHA_T, BETA_S


# ── Linear EOS ──────────────────────────────────────────────────────

def compute_density(T, S, physics, eos_type="linear"):
    """
    Compute seawater density.

    Args:
        T: temperature field [C], shape (nx, ny, nz) or (nx, ny)
        S: salinity field [psu], same shape as T
        physics: PhysicsConfig (uses T_ref, S_ref)
        eos_type: "linear" (default) or "unesco"

    Returns: density [kg/m^3], same shape as T
    """
    if eos_type == "linear":
        return _density_linear(T, S, physics)
    elif eos_type == "unesco":
        return _density_unesco(T, S)
    else:
        raise ValueError(f"Unknown eos_type: {eos_type}")


def _density_linear(T, S, physics):
    """Linearized Boussinesq EOS."""
    return RHO_0 * (
        1.0 - ALPHA_T * (T - physics.T_ref) + BETA_S * (S - physics.S_ref)
    )


# ── UNESCO nonlinear EOS ────────────────────────────────────────────
# Reference: Fofonoff & Millard (1983), UNESCO Tech. Papers in Marine
# Science No. 44. Valid at atmospheric pressure (0 dbar gauge).

# Density of pure water at 1 atm
_RHO_PURE = (
    999.842594
    + 6.793952e-2 * np.array(0)  # placeholder, see _rho_smow
)


def _rho_smow(T):
    """Density of Standard Mean Ocean Water (pure water) [kg/m^3].

    Polynomial from UNESCO 1980 EOS.
    """
    return (
        999.842594
        + 6.793952e-2 * T
        - 9.095290e-3 * T**2
        + 1.001685e-4 * T**3
        - 1.120083e-6 * T**4
        + 6.536332e-9 * T**5
    )


def _b(T):
    """Salinity coefficients B(T) for UNESCO EOS."""
    return (
        8.24493e-1
        - 4.0899e-3 * T
        + 7.6438e-5 * T**2
        - 8.2467e-7 * T**3
        + 5.3875e-9 * T**4
    )


def _c(T):
    """Salinity coefficient C(T) for UNESCO EOS."""
    return (
        -5.72466e-3
        + 1.0227e-4 * T
        - 1.6546e-6 * T**2
    )


def _d(T):
    """Salinity coefficient D(T) for UNESCO EOS."""
    return 4.8314e-4


def _density_unesco(T, S):
    """UNESCO 1980 nonlinear equation of state at 1 atm.

    rho(T,S) = rho_smow(T)
               + B(T)*S
               + C(T)*S^(3/2)
               + D(T)*S^2

    Valid for T in [-2, 40] C, S in [0, 42] psu.

    Args:
        T: temperature [C], any shape
        S: salinity [psu], same shape as T

    Returns: density [kg/m^3]
    """
    T = np.asarray(T, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)

    smow = _rho_smow(T)
    S_sqrt = np.sqrt(np.maximum(S, 0.0))  # guard against negative S

    rho = smow + _b(T) * S + _c(T) * S_sqrt * S + _d(T) * S**2
    return rho


# ── Buoyancy and density anomaly ────────────────────────────────────

def compute_buoyancy(T, S, physics, eos_type="linear"):
    """
    Buoyancy perturbation: b = -g * (rho - rho_0) / rho_0

    For linear EOS: b = g * [alpha*(T-T_ref) - beta*(S-S_ref)]

    Args:
        T: temperature [C]
        S: salinity [psu]
        physics: PhysicsConfig
        eos_type: "linear" or "unesco"

    Returns: buoyancy [m/s^2], same shape as T
    """
    from config import G_EARTH
    if eos_type == "linear":
        return G_EARTH * (
            ALPHA_T * (T - physics.T_ref) - BETA_S * (S - physics.S_ref)
        )
    elif eos_type == "unesco":
        rho = _density_unesco(T, S)
        return -G_EARTH * (rho - RHO_0) / RHO_0
    else:
        raise ValueError(f"Unknown eos_type: {eos_type}")


def density_anomaly(T, S, physics, eos_type="linear"):
    """rho' = rho - rho_0 [kg/m^3]. Convenience for pressure computation."""
    if eos_type == "linear":
        return RHO_0 * (
            -ALPHA_T * (T - physics.T_ref) + BETA_S * (S - physics.S_ref)
        )
    elif eos_type == "unesco":
        return _density_unesco(T, S) - RHO_0
    else:
        raise ValueError(f"Unknown eos_type: {eos_type}")


# ── Self-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== EOS Comparison: Linear vs UNESCO ===")
    print()

    test_cases = [
        (15.0, 35.0, "Surface subtropical"),
        (5.0, 34.5, "Subpolar surface"),
        (25.0, 35.0, "Tropical surface"),
        (2.0, 34.0, "Deep water (~2000m)"),
        (-1.5, 34.5, "Polar surface"),
    ]

    class _P:
        T_ref = 15.0
        S_ref = 35.0

    p = _P()

    print(f"{'Case':<25} {'T(C)':>6} {'S(psu)':>7} "
          f"{'rho_lin':>10} {'rho_unesco':>12} {'diff':>8}")
    print("-" * 72)

    for T, S, label in test_cases:
        rho_lin = _density_linear(T, S, p)
        rho_un = _density_unesco(T, S)
        diff = rho_un - rho_lin
        print(f"{label:<25} {T:6.1f} {S:7.1f} "
              f"{rho_lin:10.3f} {rho_un:12.3f} {diff:8.3f}")

    print()
    print("rho_0 =", RHO_0, "kg/m^3")
    print()
    print("Key checks:")
    print("  - At T=15, S=35: linear gives exactly rho_0")
    print("  - UNESCO at T=15, S=35 should be close to 1025.97")
    print("  - Cold/salty water should be denser in both EOS")
