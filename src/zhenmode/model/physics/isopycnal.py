"""Canonical closures definitions; legacy operations unchanged."""
from zhenmode.model.numerics.backend import jnp
from zhenmode.model.numerics.horizontal import (
    _dealias_h_fd,
    _divergence_conservative_3d,
    _gradient_conservative_3d,
)
from zhenmode.model.numerics.vertical import _d_dz, _fill_ghost_bottom, _iface_flux_divergence
from zhenmode.model.physics.eos import _density_anomaly

_GM_RHOZ_FLOOR = 1.0e-5
_REDI_CFL_TARGET = 0.4


def _isopycnal_slope(state, p):
    """Isopycnal slope S = (S_x, S_y) = -grad_h(rho') / d(rho')/dz.

    With z increasing downward (index 0 = surface) a stable column has d(rho')/dz > 0
    and the slope points down the horizontal density gradient. The denominator is
    floored at _GM_RHOZ_FLOOR with its sign preserved, and the slope is multiplied
    by the DM95 taper sigma in [0, 1], keeping |S| <= gm_slope_max while decaying the
    closure on steep or weakly stratified faces. (D17)

    Returns (S_x, S_y), each (nx, ny, nz), masked to wet points.
    """
    rho_prime = _density_anomaly(state.T, state.S, p) * p.wet_mask_z
    # Seafloor no-flux fill (D8): without it the bottom wet layer's _d_dz sees
    # the masked 0 below it, i.e. an INVERTED column, and the closure reads
    # "convective" (slope at its cap) over every sill.
    rho_fill = _fill_ghost_bottom(rho_prime, p)
    drho_dx, drho_dy = _gradient_conservative_3d(rho_prime, p)
    drho_dz = _d_dz(rho_fill, p)
    # Floor the denominator: stable stratification only; unstable columns
    # (drho_dz<0) get the floor magnitude with their sign preserved so the
    # bolus does not reverse in convective patches.
    denom = jnp.where(jnp.abs(drho_dz) < _GM_RHOZ_FLOOR,
                      jnp.sign(drho_dz) * _GM_RHOZ_FLOOR + 1e-30, drho_dz)
    S_x = -drho_dx / denom
    S_y = -drho_dy / denom
    # Danabasoglu-McWilliams (1995) slope taper: sigma = 1/(1+(|S|/S_lim)^4),
    # full GM in the stratified interior and a suppressed closure on steep or
    # weakly stratified faces. A tanh CLIP is wrong here: it SATURATES the flux
    # at kappa*S_lim^2 instead of decaying it. (D17)
    s_max = p.gm_slope_max
    S2 = S_x * S_x + S_y * S_y
    s4 = (s_max * s_max) ** 2
    sigma = 1.0 / (1.0 + (S2 * S2) / s4)
    S_x = sigma * S_x
    S_y = sigma * S_y
    return S_x * p.wet_mask_z, S_y * p.wet_mask_z

def _redi_skew_flux_tendency(tracer, S_x, S_y, p, kappa=None):
    """Isopycnal skew-flux residual tendency for one tracer.

    The STANDARD GM/Redi form (Griffies 1998): the bolus-transport +
    isoneutral-diffusion pair is recast as a single skew-flux tensor whose vertical
    term is DIFFUSIVE (CFL ~ kappa*|S|^2*dt/dz^2) rather than advective
    (CFL ~ |w*|*dt/dz), so it stays stable in a thin surface layer. (D18)

    Args:
        tracer: (nx, ny, nz) T or S.
        S_x, S_y: (nx, ny, nz) isopycnal slopes from _isopycnal_slope.
        kappa: diffusivity [m^2/s]; if None, reads p.kappa_redi. Pass p.kappa_gm
            explicitly to use this same operator for the GM closure.
    Returns: (nx, ny, nz) tendency dC/dt, dealiased and land-masked.
    Zero when the effective kappa <= 0.
    """
    if kappa is None:
        kappa = p.kappa_redi
    if kappa <= 0.0:
        return jnp.zeros_like(tracer)
    k = kappa
    # Seafloor no-flux fill (D8) before differentiating: the ghost layers hold
    # T_ref = 15 C forever and would otherwise leak a spurious seafloor heat
    # flux into the bottom wet layer.
    tracer = _fill_ghost_bottom(tracer, p)
    dC_dx, dC_dy = _gradient_conservative_3d(tracer * p.wet_mask_z, p)
    dC_dz = _d_dz(tracer, p)
    # Horizontal skew flux F_h = -k * S * dC/dz  (x,y components)
    Fx = -k * S_x * dC_dz
    Fy = -k * S_y * dC_dz
    tend_h = -_divergence_conservative_3d(Fx, Fy, p)
    # Vertical skew flux in INTERFACE flux form (MOM6/NEMO): the flux lives on
    # interfaces k+1/2 with the interface slope S(k+1/2) = 0.5*(S[k]+S[k+1]), so
    # cell k sees a compact 3-point stencil, exactly diffusive
    # (negative-semidefinite for the |S|^2 term) with zero flux at the material
    # top/bottom. The node-flux form it replaced decoupled the even/odd
    # sublattices in z. (D18)
    Cm = tracer[..., :-1]                        # upper cell value
    Cp = tracer[..., 1:]                         # lower cell value
    dC_dz_iface = (Cp - Cm) / p.dz_iface         # (nx, ny, nz-1)
    S_x_i = 0.5 * (S_x[..., :-1] + S_x[..., 1:])
    S_y_i = 0.5 * (S_y[..., :-1] + S_y[..., 1:])
    dC_dx_i = 0.5 * (dC_dx[..., :-1] + dC_dx[..., 1:])
    dC_dy_i = 0.5 * (dC_dy[..., :-1] + dC_dy[..., 1:])
    S2_i = S_x_i * S_x_i + S_y_i * S_y_i
    # Explicit-CFL cap on the vertical skew diffusivity: the |S|^2 term is an
    # explicit interface diffusion with D_v = k*|S|^2, whose two-cell Gershgorin
    # bound is dt*D_v*(1/dz_k + 1/dz_k1)/dz_iface <= ~2 (RK2). Only thin-layer
    # steep-front corners are clipped (MOM6-style dt-dependent limiting). (D18)
    dzu = p.dz_node[..., :-1]                    # upper cell thickness h_k
    dzl = p.dz_node[..., 1:]                     # lower cell thickness h_k1
    D_v_max = _REDI_CFL_TARGET * p.dz_iface / (p.dt * (1.0 / dzu + 1.0 / dzl))
    S2_eff = jnp.minimum(S2_i, D_v_max / k)
    Fz_i = -k * (S_x_i * dC_dx_i + S_y_i * dC_dy_i + S2_eff * dC_dz_iface)
    # No isopycnal transport crosses the surface, the seafloor or a land wall:
    # zero the flux on any interface with a dry neighbour (the ghost fill above
    # already removed the T_ref reservoir).
    wet_iface_f = (p.wet_mask_z[..., :-1] > 0.5) & (p.wet_mask_z[..., 1:] > 0.5)
    Fz_i = jnp.where(wet_iface_f, Fz_i, 0.0)
    # The material top/bottom boundaries are NOT Fz_i[0]/Fz_i[-1] (those are
    # INTERIOR interfaces; zeroing them would decouple the surface/bottom nodes
    # from the vertical skew transport). The padded array F[0..nz] carries
    # F[0] = F[nz] = 0, so tend_v[k] = (Fz_i[k-1]-Fz_i[k])/dz_node. (D18)
    tend_v = _iface_flux_divergence(Fz_i, p)
    tend = tend_h + tend_v
    tend = _dealias_h_fd(tend, p)
    return tend * p.wet_mask_z

def _isopycnal_closure(state, p):
    """GM + Redi isopycnal (eddy) tendencies for T and S.

    GM bolus transport and Redi isopycnal diffusion are two halves of the SAME skew
    flux (Griffies 1998), so both use the single operator
    _redi_skew_flux_tendency: kappa_gm the adiabatic (GM) diffusivity, kappa_redi
    the dissipative (Redi) one. Because they are the same operator, enabling BOTH
    sets the closure strength to kappa_gm + kappa_redi -- do not enable both. (D18)

    Slopes are diagnosed once and shared. Returns (gm_T, gm_S, redi_T, redi_S); a
    term is the scalar 0.0 when its kappa is 0, so the caller's sum is unaffected.
    """
    if p.kappa_gm <= 0.0 and p.kappa_redi <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    S_x, S_y = _isopycnal_slope(state, p)
    if p.kappa_gm > 0.0:
        gm_T = _redi_skew_flux_tendency(state.T, S_x, S_y, p, kappa=p.kappa_gm)
        gm_S = _redi_skew_flux_tendency(state.S, S_x, S_y, p, kappa=p.kappa_gm)
    else:
        gm_T = gm_S = 0.0
    if p.kappa_redi > 0.0:
        redi_T = _redi_skew_flux_tendency(state.T, S_x, S_y, p)
        redi_S = _redi_skew_flux_tendency(state.S, S_x, S_y, p)
    else:
        redi_T = redi_S = 0.0
    return gm_T, gm_S, redi_T, redi_S
