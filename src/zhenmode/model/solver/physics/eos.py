"""Explicit default linear EOS and opt-in CT/SR reference-salinity thermodynamics."""

from zhenmode.model.config import ALPHA_T, BETA_S, C_P, CP0_TEOS10, RHO_0


def heat_capacity(params):
    """CT represents potential enthalpy/CP0; legacy temperature retains C_P."""
    if getattr(params, 'thermodynamics', 'linear') == 'teos10_reference':
        return CP0_TEOS10
    return C_P


def _density_anomaly(T, S, p):
    """ρ-ρ0: default linear relation, or explicit CT/SR at supplied dbar pressure."""
    if getattr(p, 'thermodynamics', 'linear') == 'teos10_reference':
        from zhenmode.model.solver.physics.teos10 import density
        return density(S, T, p.eos_pressure_dbar) - RHO_0
    return RHO_0 * (-ALPHA_T * (T - p.T_ref) + BETA_S * (S - p.S_ref))
