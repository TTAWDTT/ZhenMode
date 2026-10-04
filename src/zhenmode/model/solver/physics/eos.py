"""Linear seawater density anomaly with explicit reference temperature and salinity."""

from zhenmode.model.config import ALPHA_T, BETA_S, RHO_0


def _density_anomaly(T, S, p):
    """rho' = rho - rho_0 from the linear equation of state.

    The retired regional solver carried a UNESCO 1980 nonlinear branch
    behind an ``eos_type`` switch. It was never ported here -- the branch
    was a no-op stub returning this same expression -- so the switch is
    gone and the EOS is unambiguously linear. The convective-adjustment
    and buoyancy diagnostics are calibrated against this form.
    """
    return RHO_0 * (-ALPHA_T * (T - p.T_ref) + BETA_S * (S - p.S_ref))
