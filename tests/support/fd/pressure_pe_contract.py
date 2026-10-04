"""Flat-bed instantaneous pressure/first-moment controls; no full-step energy claim."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from ocean_solver.config.definitions import ALPHA_T, G_EARTH, RHO_0
from ocean_solver.dynamics.pressure import _compute_pressure_gradient
from ocean_solver.dynamics.transport import (
    _advection_scalar,
    _face_transport_divergence,
    _layer_face_transports,
    _vertical_transport_iface,
)


def fixture():
    """8x3 spherical-metric columns, four nonuniform fixed FD nodes, flat bed."""
    depth = np.array([0., 5., 20., 50.])
    h = np.array([2.5, 10., 22.5, 15.])
    cosine = np.cos(np.deg2rad([-30., 0., 30.]))
    dx = np.broadcast_to(1000. * cosine[None, :], (8, 3)).copy()
    wet = np.ones((8, 3, 4))
    p = SimpleNamespace(ny=3, wet_mask_z=jnp.asarray(wet), cos_lat=jnp.asarray(cosine),
                        inv_dx=jnp.asarray(1. / dx[..., None]), inv_dy=1. / 1000.,
                        dz_node=jnp.asarray(h), dz_3d=jnp.asarray(np.diff(depth)),
                        T_ref=15., S_ref=35., monotone_adv=True, fct_adv=False)
    return p, dx * 1000., depth


def evaluate(rho, u, v, p, area, depth):
    """W_P + dE_rho/dt at eta=0, using actual pressure and donor scalar operators.

    E_rho = -g sum A*d_k*(h0_k+eta*delta_k0)*rho'_k (J).
    Its conjugate is -g*d_k; rho' is kg/m3. This is nodal first-moment PE,
    not an established physical centroid quadrature for a moving top control.
    """
    if not np.all(np.asarray(p.wet_mask_z) == 1.) or depth[0] != 0.:
        raise ValueError('only all-wet flat-bed z0=0 oracle is supported')
    rho, u, v = map(jnp.asarray, (rho, u, v))
    state = SimpleNamespace(T=p.T_ref-rho/(RHO_0*ALPHA_T),
                            S=jnp.full_like(rho, p.S_ref), eta=jnp.zeros(rho.shape[:2]))
    ax, ay = _compute_pressure_gradient(state, p)
    h = np.asarray(p.dz_node) * np.asarray(p.wet_mask_z)
    work = float(np.sum(RHO_0 * area[..., None] * h * (u*ax + v*ay)))
    faces = _layer_face_transports(u, v, p)
    vertical = _vertical_transport_iface(u, v, p, face_transport=faces)
    tendency, surface_exchange = _advection_scalar(
        rho, u, v, vertical, p, return_boundary=True, face_transport=faces)
    # Fixed first-moment diagnostic: the top node is at depth zero.
    rate = np.asarray(h * tendency).copy()
    rate[..., 0] -= np.asarray(surface_exchange)
    pe_rate = float(-G_EARTH * np.sum(area[..., None] * depth * rate))
    f = np.asarray(vertical)[..., 1:-1]
    r = np.asarray(rho)
    donor = np.where(f > 0., r[..., :-1], r[..., 1:])
    centered = .5 * (r[..., :-1]+r[..., 1:])
    centered_flux = f * centered
    zero = np.zeros_like(r[..., :1])
    vertical_centered_rate = -(np.concatenate((centered_flux, zero), axis=-1)
                              - np.concatenate((zero, centered_flux), axis=-1))
    # Horizontal density transport has zero global first-moment power here:
    # d_k is constant on each layer and all horizontal boundaries are closed.
    center_pe = float(-G_EARTH * np.sum(area[..., None] * depth * vertical_centered_rate))
    donor_defect = float(-G_EARTH * np.sum(
        area[..., None] * np.diff(depth) * f * (donor-centered)))
    absolute = float(np.sum(RHO_0*area[..., None]*h*(np.abs(u*ax)+np.abs(v*ay)))
                     + G_EARTH*np.sum(area[..., None]*depth*np.abs(rate))
                     + G_EARTH*np.sum(area[..., None]*depth*np.abs(vertical_centered_rate))
                     + G_EARTH*np.sum(area[..., None]*np.diff(depth)*np.abs(f*(donor-centered))))
    bound = float(4096*np.finfo(np.float64).eps*max(1., absolute))
    return dict(pressure_work_W=work, donor_PE_rate_W=pe_rate,
                centered_oracle_PE_rate_W=center_pe, donor_pairing_residual_W=work+pe_rate,
                centered_pairing_residual_W=work+center_pe, donor_flux_defect_W=donor_defect,
                defect_reconstruction_residual_W=work+pe_rate-donor_defect,
                arithmetic_bound_W=bound,
                eta_rate_max_m_s=float(np.max(np.abs(np.asarray(vertical)[..., 0]))),
                density_stock_rate_kg_s=float(np.sum(area[..., None]*rate)),
                north_exterior_max_m2_s=float(np.max(np.abs(np.asarray(faces[1])[:, -1]))),
                bed_exterior_max_m_s=float(np.max(np.abs(np.asarray(vertical)[..., -1]))),
                top_PE_boundary_W=0.,  # d_0=0 and relative top tracer flux=0
                column_divergence_max_m_s=float(np.max(np.abs(np.asarray(
                    _face_transport_divergence(jnp.sum(faces[0], axis=-1),
                                               jnp.sum(faces[1], axis=-1), p))))))


def cases():
    p, area, depth = fixture()
    phase = 2*np.pi*np.arange(8)[:, None, None]/8
    profile = np.array([1., -.25, 0., 0.])  # sum h*u=0, eta_dot=0
    u = np.broadcast_to(.01*np.sin(phase)*profile, (8, 3, 4)).copy()
    v = np.zeros_like(u)
    stable = np.broadcast_to(np.array([0., .2, .8, 2.]), u.shape).copy()
    variable = stable + np.broadcast_to(.1*np.cos(phase)*np.array([1., 2., 0., 0.]), u.shape)
    return {name: evaluate(r, a, v, p, area, depth) for name, r, a in (
        ('rest_stable', stable, np.zeros_like(u)),
        ('circulation_uniform_stable', stable, u),
        ('layered_horizontal_exchange', variable, u),
) }
