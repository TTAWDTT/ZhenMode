"""Mechanically preserved FD projection implementation."""
from config import G_EARTH

from .backend import jax, jnp
from .horizontal import _gradient_conservative_3d
from .transport import _column_divergence, _vertical_transport_iface


def _column_projection_diagonal(p):
    """Exact diagonal of -A*B*G3, including closed wet faces and walls."""
    wet = p.wet_mask_z
    positive_x = wet * jnp.roll(wet, -1, axis=0)
    negative_x = wet * jnp.roll(wet, 1, axis=0)
    diagonal_x = 0.25 * p.inv_dx[..., :1] ** 2 * (
        (positive_x - negative_x) ** 2 + positive_x ** 2 + negative_x ** 2)
    if p.nx <= 2:
        diagonal_x = jnp.zeros_like(wet)
    cosine = p.cos_lat[None, :, None]
    cosine_north = jnp.roll(cosine, -1, axis=1)
    cosine_south = jnp.roll(cosine, 1, axis=1)
    positive_y = (wet * jnp.roll(wet, -1, axis=1)).at[:, -1, :].set(0.)
    negative_y = (wet * jnp.roll(wet, 1, axis=1)).at[:, 0, :].set(0.)
    positive_y = positive_y * 0.5 * (cosine + cosine_north)
    negative_y = negative_y * 0.5 * (cosine + cosine_south)
    diagonal_y = 0.25 * p.inv_dy ** 2 * (
        (positive_y - negative_y) ** 2 / cosine ** 2
        + positive_y ** 2 / (cosine * cosine_north)
        + negative_y ** 2 / (cosine * cosine_south))
    area = p.dx_2d * p.dy * p.wet_mask
    return area * jnp.sum((diagonal_x + diagonal_y) * p.dz_node, axis=-1)

def projection_config(params):
    """Effective immutable solve settings for machine-readable provenance."""
    tolerance = max(params.projection_rtol or 1e-12,
                    32. * float(jnp.finfo(params.wet_mask_z.dtype).eps))
    return {"enabled": bool(params.project_adv_vel), "niter": params.projection_niter,
            "rtol": tolerance, "preconditioner": params.projection_preconditioner,
            "niter_source": params.projection_niter_source,
            "max_refinements": params.projection_max_refinements,
            "transport_rtol": max(1e-10, 32. * float(jnp.finfo(params.wet_mask_z.dtype).eps)),
            "refinement_atol_scale": "original_rhs_l2"}

def _project_column_divergence(u, v, p, dt, n_iter=None):
    """Return (u, v) with the column-integrated horizontal divergence removed.

    The Euler correction (u, v) -= dt*g*grad(psi) of a surface-pressure potential
    that solves the area-weighted Poisson problem with the EXACT column divergence
    (not a constant-H 2D one, which leaves ~40% of the residual at coastlines).
    The 3D wet-face gradient is the volume-weighted negative adjoint of that
    constraint; a broadcast 2D gradient is not equivalent at bottom steps.
    CG stops at a dtype-aware tolerance or the fixed iteration cap. Bounded
    reprojection checks the actual transport, not the recursive CG residual.
    Correction solves retain an absolute floor from the original RHS. (D36)
    """
    if n_iter is None:
        n_iter = p.projection_niter
    g = G_EARTH
    wm = p.wet_mask
    area = p.dx_2d * p.dy * wm
    col_div = _column_divergence(u, v, p)
    rhs = -(col_div / (dt * g)) * area

    def _matvec(psi):
        gx, gy = _gradient_conservative_3d(psi[:, :, None], p)
        return -_column_divergence(gx, gy, p) * area

    tolerance = max(p.projection_rtol or 1e-12, 32. * jnp.finfo(u.dtype).eps)
    preconditioner = None
    if p.projection_preconditioner == "jacobi":
        inverse_diagonal = p.projection_inv_diagonal
        if inverse_diagonal is None:
            diagonal = _column_projection_diagonal(p)
            inverse_diagonal = 1. / jnp.where(diagonal > 0., diagonal, 1.)
        def preconditioner(residual):
            return residual * inverse_diagonal
    psi, _ = jax.scipy.sparse.linalg.cg(_matvec, rhs, tol=tolerance,
                                        maxiter=n_iter, M=preconditioner)

    def correct(velocities, potential):
        gradient = _gradient_conservative_3d(potential[..., None], p)
        return tuple(velocity - (dt * g) * derivative
                     for velocity, derivative in zip(velocities, gradient, strict=True))

    corrected = correct((u, v), psi)
    if p.projection_max_refinements == 0:
        return corrected

    def transport_norm_squared(velocities):
        transport = _vertical_transport_iface(*velocities, p)[..., 0].astype(jnp.float64)
        area64 = p.dx_2d.astype(jnp.float64) * p.dy * wm
        return jnp.sum(transport ** 2 * area64)

    transport_tolerance = max(1e-10, 32. * jnp.finfo(u.dtype).eps)
    threshold = transport_tolerance ** 2 * transport_norm_squared((u, v))
    absolute_floor = tolerance * jnp.linalg.norm(rhs)

    def refine(velocities):
        residual = -(_column_divergence(*velocities, p) / (dt * g)) * area
        delta, _ = jax.scipy.sparse.linalg.cg(_matvec, residual, tol=tolerance,
                    atol=absolute_floor, maxiter=n_iter, M=preconditioner)
        return correct(velocities, delta)

    for refinement in range(p.projection_max_refinements):
        corrected = jax.lax.cond(transport_norm_squared(corrected) > threshold,
                                 refine, lambda velocities: velocities, corrected)
    return corrected
