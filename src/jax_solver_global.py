"""
Global Finite-Difference Ocean Solver — hydrostatic primitive equations, JAX.

Numerics:
  - 2nd-order finite differences on a global lat-lon grid (lon-periodic),
  - spherical metric factors (dx = R*cos(lat)*dlon, varies with latitude),
  - a real wet_mask for no-flux land boundaries,
  - full 2D Coriolis f = 2*Omega*sin(lat).

Convention (shared with grid.py):
  - 3D fields: (nx, ny, nz), axis 0 = lon (periodic), axis 1 = lat, axis 2 = z
  - 2D fields: (nx, ny)
  - z negative downward, z=0 at surface

Every FD operator is a pure function of (field, params); `params` carries the
precomputed metric fields. The reasoning behind each closure, limiter and
subcycle count — with the measurements that settled it — lives in
docs/decisions.md, keyed by the D-numbers referenced below. Keep it there:
this file is the solver, not the lab notebook.
"""
import os

# On-demand GPU memory by default: shared desktop GPUs and fine grids are
# less likely to hit CUDA OOM. Override with the env var when benchmarking.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update('jax_enable_x64', True)
from collections import namedtuple

import jax.numpy as jnp
import jax.scipy.sparse.linalg
import numpy as np

from config import ALPHA_T, BETA_S, C_P, G_EARTH, OMEGA, R_EARTH, RHO_0
from grid import nodal_control_thickness

# ── State ──────────────────────────────────────────────────────────
JaxStateG = namedtuple('JaxStateG', ['u', 'v', 'T', 'S', 'eta', 'ice'])
# Older five-field callers remain valid; a new run starts with no ice.
JaxStateG.__new__.__defaults__ = (0.0,)


# ── FD solver parameters ───────────────────────────────────────────
# Metric + grid fields for the FD operators. Built once from a GlobalOceanGrid.
FDParams = namedtuple('FDParams', [
    # Spherical metric
    'dx_2d',            # (nx, ny) zonal spacing [m] = R*cos(lat)*dlon
    'dy',               # scalar meridional spacing [m]
    'cos_lat',          # (ny,) cos(lat)
    'inv_dx',           # (nx, ny) 1/dx_2d
    'inv_dy',           # scalar 1/dy
    'inv_dx2',          # (nx, ny) 1/dx_2d^2
    'inv_dy2',          # scalar 1/dy^2
    # Coriolis
    'f',                # (nx, ny) full 2D Coriolis
    # Land
    'wet_mask',         # (nx, ny) 1=ocean, 0=land
    'wet_mask_z',       # (nx, ny, nz) TRUE vertical wet mask: 1 where layer is
                        # above the seafloor, 0 below (ghost water excluded).
                        # Used in pressure integration to kill the spurious PGF
                        # at steep topography (ghost-water-column bug fix).
    'interior_mask_z',  # (nx, ny, 1) 1 in the interior, 0 on the N/S boundary
                        # rows (j=0, j=ny-1). Enforces the no-flux wall: the
                        # normal (meridional) velocity v is zeroed here so no
                        # flow crosses the closed N/S truncation wall.
    # Vertical grid (non-uniform z-levels)
    'dz_denom_interior', 'dz_bnd_top', 'dz_bnd_bot',
    'd2z_hm', 'd2z_hp', 'd2z_denom', 'd2z_h0_top', 'd2z_h0_bot',
    'dz_3d', 'dz_surface', 'dz_iface', 'dz_node',
    'surface_mask', 'bottom_mask',   # (1,1,nz)
    # Dimensions
    'nx', 'ny', 'nz',
])


def make_fd_params(grid, column_geometry='legacy'):
    """Build FDParams from a GlobalOceanGrid (grid.py).

    Precomputes all metric inverse fields so the FD operators are pure
    array arithmetic (no division inside the hot loop).
    """
    if column_geometry not in {'legacy', 'nodal_dual_v1'}:
        raise ValueError("column_geometry must be legacy or nodal_dual_v1")
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx_2d = jnp.array(grid.dx_2d)                # (nx, ny)
    dy = float(grid.dy)
    cos_lat = jnp.array(grid.cos_lat)            # (ny,)

    inv_dx = (1.0 / dx_2d)[:, :, None]           # (nx, ny, 1) for 3D broadcast
    inv_dy = 1.0 / dy
    inv_dx2 = (1.0 / dx_2d ** 2)[:, :, None]     # (nx, ny, 1)
    inv_dy2 = inv_dy ** 2

    f = jnp.array(grid.f)                        # (nx, ny)
    wet_mask = jnp.array(grid.wet_mask)          # (nx, ny)
    wet_mask_3d = wet_mask[:, :, None]           # (nx, ny, 1)
    # True 3D wet mask (layer-resolved): from the grid if it carries one,
    # else fall back to column-uniform (synthetic grids without bathymetry).
    if hasattr(grid, 'wet_mask_3d') and grid.wet_mask_3d is not None:
        wet_mask_z = jnp.array(grid.wet_mask_3d)      # (nx, ny, nz)
    else:
        wet_mask_z = jnp.broadcast_to(wet_mask_3d, (nx, ny, nz))

    # No-flux wall mask: 1 interior, 0 on the N/S boundary rows. The closed
    # truncation wall zero normal velocity here (interior_mask_z applied to v).
    interior_1d = np.ones(ny)
    interior_1d[0] = 0.0
    interior_1d[-1] = 0.0
    interior_mask = jnp.array(np.broadcast_to(interior_1d[None, :], (nx, ny)))
    interior_mask_z = interior_mask[:, :, None]

    # Vertical grid coefficients (non-uniform z-level metric terms)
    z = jnp.array(grid.z)
    dz = jnp.array(grid.dz)
    dz_3d = dz.reshape(1, 1, -1)
    dz_up = z[1:-1] - z[:-2]
    dz_dn = z[2:] - z[1:-1]
    dz_denom_interior = (dz_up + dz_dn).reshape(1, 1, -1)
    dz_bnd_top = float(z[1] - z[0])
    dz_bnd_bot = float(z[-1] - z[-2])
    hm = jnp.abs(z[:-2] - z[1:-1])
    hp = jnp.abs(z[2:] - z[1:-1])
    d2z_denom = (hm * hp * (hm + hp) / 2.0).reshape(1, 1, -1)
    d2z_hm = hm.reshape(1, 1, -1)
    d2z_hp = hp.reshape(1, 1, -1)
    d2z_h0_top = float(jnp.abs(z[1] - z[0]))
    d2z_h0_bot = float(jnp.abs(z[-1] - z[-2]))
    dz_surface = float(jnp.abs(z[0] - z[1]))
    # Interface thicknesses dz_iface[k] = |z[k]-z[k+1]|, length nz-1 (interfaces
    # k+1/2 between nodes k and k+1); used by the GM/Redi interface flux form.
    dz_iface = jnp.abs(jnp.diff(z)).reshape(1, 1, -1)
    # Node-cell thickness: the model is NODE-based (fields live on z-levels), so
    # the "cell" around node k spans halfway to each neighbour: dz_node[0] =
    # |z1-z0|, dz_node[k] = 0.5*(|zk-zk-1|+|zk+1-zk|), dz_node[-1] = |zN-1-zN-2|.
    # Length nz. Used by the interface flux form for
    # tend_v[k] = (F[k-1/2]-F[k+1/2]) / dz_node[k].
    dz_node = jnp.concatenate([
        jnp.array([float(jnp.abs(z[1] - z[0]))]),
        0.5 * (jnp.abs(jnp.diff(z))[:-1] + jnp.abs(jnp.diff(z))[1:]),
        jnp.array([float(jnp.abs(z[-1] - z[-2]))]),
    ]).reshape(1, 1, -1)

    surface_mask = jnp.zeros(nz).at[0].set(1.0).reshape(1, 1, -1)
    bottom_mask = jnp.zeros(nz).at[-1].set(1.0).reshape(1, 1, -1)

    if column_geometry == 'nodal_dual_v1':
        thickness = nodal_control_thickness(grid.z)
        wet = np.broadcast_to(np.asarray(wet_mask_z), (nx, ny, nz))
        if (not np.all((wet == 0.) | (wet == 1.)) or np.any(np.diff(wet, axis=-1) > 0.)
                or not np.array_equal(wet[..., 0], np.asarray(grid.wet_mask))):
            raise ValueError("nodal_dual_v1 requires binary, contiguous wet columns with a wet surface")
        wet_mask_z = jnp.asarray(wet)
        dz_node = jnp.asarray(thickness).reshape(1, 1, -1)
        dz_surface = float(thickness[0])
        next_wet = jnp.concatenate((wet_mask_z[..., 1:], jnp.zeros_like(wet_mask_z[..., :1])), axis=-1)
        bottom_mask = wet_mask_z * (1. - next_wet)

    return FDParams(
        dx_2d=dx_2d, dy=dy, cos_lat=cos_lat,
        inv_dx=inv_dx, inv_dy=inv_dy, inv_dx2=inv_dx2, inv_dy2=inv_dy2,
        f=f, wet_mask=wet_mask, wet_mask_z=wet_mask_z,
        interior_mask_z=interior_mask_z,
        dz_denom_interior=dz_denom_interior,
        dz_bnd_top=dz_bnd_top, dz_bnd_bot=dz_bnd_bot,
        d2z_hm=d2z_hm, d2z_hp=d2z_hp, d2z_denom=d2z_denom,
        d2z_h0_top=d2z_h0_top, d2z_h0_bot=d2z_h0_bot,
        dz_3d=dz_3d, dz_surface=dz_surface, dz_iface=dz_iface, dz_node=dz_node,
        surface_mask=surface_mask, bottom_mask=bottom_mask,
        nx=nx, ny=ny, nz=nz,
    )


# ── Horizontal FD operators (2nd-order, spherical metric, lon-periodic) ─

def _d_dx(u, p):
    """Zonal derivative du/dx, 2nd-order central, longitude-periodic.

    Land points are differentiated like any other node; the callers mask. (D2)
    """
    return (jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) * (0.5 * p.inv_dx)


def _d_dy(u, p):
    """Meridional derivative du/dy, 2nd-order central with NO-FLUX WALLS.

    The lat axis does not wrap: a mirror ghost cell (ghost = boundary value) gives
    the wall row a stable central difference with zero normal diffusive flux, and
    the normal velocity is masked to 0 there so no advective flux crosses either.
    Index 0 is the SOUTH row. (D1)
    """
    # Mirror-pad the lat axis by 1 cell each side (ghost = boundary value).
    # ndim-agnostic: pad only axis 1.
    pad = [(0, 0)] * u.ndim
    pad[1] = (1, 1)
    u_pad = jnp.pad(u, pad, mode='edge')
    return (u_pad[:, 2:] - u_pad[:, :-2]) * (0.5 * p.inv_dy)


def _divergence_conservative(ubt, vbt, p):
    """Mass-conserving horizontal divergence of the barotropic velocity.

    Flux form: a face is open only when BOTH adjacent cells are wet, and the N/S
    truncation walls are closed. The zonal term uses the per-cell inv_dx = dy/A_i
    and the meridional term the spherical form (1/cos_j)*d(v*cos_face)/dy, so the
    divergence telescopes to zero under the area weight A_ij = R^2*cos(lat)*dphi^2.
    The bare centered form leaks volume at coastlines and walls. (D2)
    """
    wm = p.wet_mask                                   # (nx, ny)
    cos_lat = p.cos_lat                               # (ny,)
    # Zonal (axis 0, periodic): face (i+1/2,j) open iff cell i and i+1 both wet.
    uface_open = wm * jnp.roll(wm, -1, axis=0)        # 1 where both sides wet
    uface = 0.5 * (ubt + jnp.roll(ubt, -1, axis=0)) * uface_open
    div_x = (uface - jnp.roll(uface, 1, axis=0)) * p.inv_dx[..., 0]
    # Meridional (axis 1, closed N/S walls): face (i,j+1/2) open iff j and j+1
    # both wet. The roll wraps j=ny-1 -> j=0 across the pole, so explicitly
    # close the two boundary faces (no flow through the truncation walls).
    vface_open = wm * jnp.roll(wm, -1, axis=1)
    vface_open = vface_open.at[:, -1].set(0.0)        # north wall closed
    vface = 0.5 * (vbt + jnp.roll(vbt, -1, axis=1)) * vface_open
    cos_face = 0.5 * (cos_lat + jnp.roll(cos_lat, -1))   # cos at face j+1/2
    fcos = vface * cos_face[None, :]                     # mass flux v*cos(face)
    fcos_in = jnp.roll(fcos, 1, axis=1)                  # flux at face j-1/2
    fcos_in = fcos_in.at[:, 0].set(0.0)                  # south wall closed
    div_y = (fcos - fcos_in) * p.inv_dy / cos_lat[None, :]
    return div_x + div_y


def _gradient_conservative(eta, p):
    """Adjoint of _divergence_conservative, for the free-surface pressure gradient.

    Face differences weighted by the cell inv_dx (meridionally by the face cos and
    divided by the cell cos). The centered _d_dx/_d_dy is NOT the adjoint: pairing
    it with the conservative divergence injects energy into the free mode. (D3)
    """
    wm = p.wet_mask
    cos_lat = p.cos_lat                               # (ny,)
    inv_dx = p.inv_dx[..., 0]                          # (nx, ny)
    # Zonal (axis 0, periodic): open face iff both neighbors wet.
    open_xp = wm * jnp.roll(wm, -1, axis=0)            # face (i+1/2,j)
    open_xm = wm * jnp.roll(wm, 1, axis=0)             # face (i-1/2,j)
    d_eta_xp = (jnp.roll(eta, -1, axis=0) - eta) * open_xp   # eta_{i+1}-eta_i
    d_eta_xm = (eta - jnp.roll(eta, 1, axis=0)) * open_xm    # eta_i-eta_{i-1}
    grad_x = inv_dx * 0.5 * (d_eta_xp + d_eta_xm)
    # Meridional (axis 1, closed walls): adjoint of the spherical divergence,
    # face cos weighted, divided by the cell cos.
    open_yp = wm * jnp.roll(wm, -1, axis=1)
    open_ym = wm * jnp.roll(wm, 1, axis=1)
    # Close the two boundary faces (adjoint of the closed truncation walls).
    open_yp = open_yp.at[:, -1].set(0.0)               # north wall
    open_ym = open_ym.at[:, 0].set(0.0)                # south wall
    cos_face_p = 0.5 * (cos_lat + jnp.roll(cos_lat, -1))   # face j+1/2
    cos_face_m = 0.5 * (cos_lat + jnp.roll(cos_lat, 1))    # face j-1/2
    d_eta_yp = (jnp.roll(eta, -1, axis=1) - eta) * open_yp * cos_face_p[None, :]
    d_eta_ym = (eta - jnp.roll(eta, 1, axis=1)) * open_ym * cos_face_m[None, :]
    grad_y = (p.inv_dy / cos_lat[None, :]) * 0.5 * (d_eta_yp + d_eta_ym)
    return grad_x, grad_y


def _column_divergence(u, v, p):
    """Column-integrated horizontal divergence (nx, ny).

    SUM_k div_h[k]*dz_node[k], built from the same per-layer face-flux operator the
    advection budget uses, so a zero here is exactly the discretely
    divergence-free condition the tracer flux operator needs. (D13)
    """
    return jnp.sum(_divergence_h(u, v, p) * p.dz_node, axis=-1) * p.wet_mask


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


def _gradient_conservative_3d(field, p):
    """Masked, cos(lat)-weighted adjoint gradient, face-gated like the divergence.

    No flux reaches into land zeros across a coastline. (D4)
    """
    wm = p.wet_mask_z                                 # (nx, ny, nz)
    cos_lat = p.cos_lat                               # (ny,)
    inv_dx = p.inv_dx[..., 0:1]                        # (nx, ny, 1)
    # Zonal (axis 0, periodic): open face iff both neighbors wet.
    open_xp = wm * jnp.roll(wm, -1, axis=0)
    open_xm = wm * jnp.roll(wm, 1, axis=0)
    d_fp = (jnp.roll(field, -1, axis=0) - field) * open_xp
    d_fm = (field - jnp.roll(field, 1, axis=0)) * open_xm
    grad_x = inv_dx * 0.5 * (d_fp + d_fm)
    # Meridional (axis 1, closed walls): adjoint of the spherical divergence.
    open_yp = wm * jnp.roll(wm, -1, axis=1)
    open_ym = wm * jnp.roll(wm, 1, axis=1)
    open_yp = open_yp.at[:, -1, :].set(0.0)           # north wall
    open_ym = open_ym.at[:, 0, :].set(0.0)            # south wall
    cos_face_p = 0.5 * (cos_lat + jnp.roll(cos_lat, -1))   # face j+1/2
    cos_face_m = 0.5 * (cos_lat + jnp.roll(cos_lat, 1))    # face j-1/2
    shp = [1, p.ny, 1]
    d_yp = (jnp.roll(field, -1, axis=1) - field) * open_yp * cos_face_p.reshape(shp)
    d_ym = (field - jnp.roll(field, 1, axis=1)) * open_ym * cos_face_m.reshape(shp)
    grad_y = (p.inv_dy / cos_lat[None, :, None]) * 0.5 * (d_yp + d_ym)
    return grad_x, grad_y


def _divergence_conservative_3d(Fx, Fy, p):
    """3D divergence whose faces are gated on BOTH adjacent cells being wet.

    The bare _d_dx/_d_dy divergence of a coastal flux reads the land zeros inside
    its stencil and injects a spurious coastal source. (D4)
    """
    wm = p.wet_mask_z                                 # (nx, ny, nz)
    cos_lat = p.cos_lat                               # (ny,)
    # Zonal (axis 0, periodic): face (i+1/2,j,k) open iff cell i and i+1 wet.
    uface_open = wm * jnp.roll(wm, -1, axis=0)
    uface = 0.5 * (Fx + jnp.roll(Fx, -1, axis=0)) * uface_open
    div_x = (uface - jnp.roll(uface, 1, axis=0)) * p.inv_dx[..., 0:1]
    # Meridional (axis 1, closed walls): face (i,j+1/2,k) open iff j,j+1 wet.
    vface_open = wm * jnp.roll(wm, -1, axis=1)
    vface_open = vface_open.at[:, -1, :].set(0.0)    # north wall closed
    vface = 0.5 * (Fy + jnp.roll(Fy, -1, axis=1)) * vface_open
    cos_face = 0.5 * (cos_lat + jnp.roll(cos_lat, -1))   # cos at face j+1/2
    shp = [1, p.ny, 1]
    fcos = vface * cos_face.reshape(shp)                  # mass flux v*cos(face)
    fcos_in = jnp.roll(fcos, 1, axis=1)
    fcos_in = fcos_in.at[:, 0, :].set(0.0)                 # south wall closed
    div_y = (fcos - fcos_in) * p.inv_dy / cos_lat[None, :, None]
    return div_x + div_y


def _laplacian_h(u, p):
    """Horizontal Laplacian on the sphere, in physical (meter) space:

        grad^2 u = 1/cos(phi) d/dphi(cos(phi) du/dphi) + 1/cos^2(phi) d2u/dlam2
                 = d2u/dy2 - (tan(phi)/R) du/dy + (1/cos^2(phi)) d2u/dx2

    Both second differences are FACE-GATED (open iff both cells wet) and the lat one
    uses the mirror ghost cell of _d_dy, so the Laplacian sees a flat profile across
    a closed face instead of the mask step. (D5)
    """
    # ∂²u/∂x²: FACE-GATED conservative form (open iff BOTH cells wet). The bare
    # central stencil reads the mask step at the coast, where ghost nodes hold
    # the T_ref sentinel, and the spike/dipole it produces drifts whole coastal
    # cells. (D5)
    wm = p.wet_mask_z                                   # (nx, ny, nz)
    d2u_dx2 = ((jnp.roll(u, -1, axis=0) - u) * wm * jnp.roll(wm, -1, axis=0)
               - (u - jnp.roll(u, 1, axis=0)) * wm * jnp.roll(wm, 1, axis=0)) * p.inv_dx2
    # ∂²u/∂y²: same wet/wet face gate in y, plus the mirror ghost cell of
    # _d_dy (ghost = boundary value), giving zero normal gradient at the N/S
    # wall: d²u/dy²|_0 = (u1 - 2u0 + u_ghost)/dy² = (u1 - u0)/dy². (D1, D5)
    pad = [(0, 0)] * u.ndim
    pad[1] = (1, 1)
    u_pad = jnp.pad(u, pad, mode='edge')
    wm_pad = jnp.pad(wm, pad, mode='edge')
    d2u_dy2 = ((u_pad[:, 2:] - u_pad[:, 1:-1]) * wm_pad[:, 2:] * wm_pad[:, 1:-1]
               - (u_pad[:, 1:-1] - u_pad[:, :-2]) * wm_pad[:, 1:-1] * wm_pad[:, :-2]) * p.inv_dy2
    # spherical metric correction: -(tanφ/R) ∂u/∂y
    # tan(lat) = sin(lat)/cos(lat); signed sin(lat) from f = 2*Omega*sin(lat)
    # (constant along longitude, so row 0 of f gives the per-row sin).
    du_dy = _d_dy(u, p)
    sin_lat_row = p.f[0, :] / (2.0 * OMEGA)           # (ny,) signed sin(lat)
    tan_over_R = (sin_lat_row / p.cos_lat) / R_EARTH   # (ny,) tan(lat)/R
    shp = [1] * u.ndim
    shp[1] = p.ny
    corr = -du_dy * tan_over_R.reshape(shp)
    return d2u_dx2 + d2u_dy2 + corr


def _horizontal_diffusion_flux(tracer, diffusivity, p):
    """Wet-face variable-coefficient diffusion with closed latitude walls."""
    wet = p.wet_mask_z
    zonal_flux = (0.5 * (diffusivity + jnp.roll(diffusivity, -1, axis=0))
                  * (jnp.roll(tracer, -1, axis=0) - tracer) * p.inv_dx
                  * wet * jnp.roll(wet, -1, axis=0))
    padding = [(0, 0), (0, 1), (0, 0)]
    next_tracer = jnp.pad(tracer, padding, mode='edge')[:, 1:, :]
    next_diffusivity = jnp.pad(diffusivity, padding, mode='edge')[:, 1:, :]
    next_wet = jnp.pad(wet, padding, mode='edge')[:, 1:, :]
    cos_face = 0.5 * (p.cos_lat + jnp.roll(p.cos_lat, -1))
    meridional_flux = (0.5 * (diffusivity + next_diffusivity)
                       * (next_tracer - tracer) * p.inv_dy
                       * wet * next_wet * cos_face[None, :, None])
    meridional_flux = meridional_flux.at[:, -1, :].set(0.)
    incoming_y = jnp.roll(meridional_flux, 1, axis=1).at[:, 0, :].set(0.)
    return ((zonal_flux - jnp.roll(zonal_flux, 1, axis=0)) * p.inv_dx
            + (meridional_flux - incoming_y) * p.inv_dy / p.cos_lat[None, :, None])


def _horizontal_tracer_diffusion(tracer, p):
    """Conservative wet-face diffusion for background and enhanced coefficients."""
    diffusivity = p.kappa_h + p.coastal_kappa_h_2d[:, :, None]
    return _horizontal_diffusion_flux(tracer, diffusivity, p)


def _horizontal_biharmonic_tracer(tracer, p):
    """Square the self-adjoint wet-face Laplacian; negative sign dissipates."""
    unit_diffusivity = jnp.ones_like(p.coastal_kappa_h_2d)[:, :, None]
    laplacian = _horizontal_diffusion_flux(tracer, unit_diffusivity, p)
    return _horizontal_diffusion_flux(laplacian, unit_diffusivity, p)


def _biharmonic_h(u, p):
    """Biharmonic grad^4 u = grad^2(grad^2 u), two face-gated Laplacians.

    The gate keeps the mask-step halo out of the inner Laplacian, so it cannot seed
    the outer one. (D5)
    """
    return _laplacian_h(_laplacian_h(u, p), p)


def _divergence_h(u, v, p):
    """Horizontal divergence from FACE-FLUX differences, as advection sees it.

    Algebraically identical to the centered difference on fully-wet nodes, but at a
    coastal wall the face-flux form measures the real inflow through the open face
    where the centered form measures the shear against the land neighbour's u = 0.
    Only the former lets the diagnosed w close the advective budget. Meridionally it
    mirror-pads the closed N/S walls and carries the same cos(face)/cos(cell)
    spherical factors as _divergence_conservative. (D6)
    """
    wm = p.wet_mask_z
    # Zonal (axis 0, periodic): face (i+1/2) velocity, open iff both wet.
    Fx = 0.5 * (u + jnp.roll(u, -1, axis=0)) * wm * jnp.roll(wm, -1, axis=0)
    div_x = (Fx - jnp.roll(Fx, 1, axis=0)) * p.inv_dx[..., 0:1]
    # Meridional (axis 1, closed N/S walls): edge-pad, face (j+1/2).
    pad = [(0, 0)] * v.ndim
    pad[1] = (1, 1)
    v_pad = jnp.pad(v, pad, mode='edge')
    wm_pad = jnp.pad(wm, pad, mode='edge')
    Fy = (0.5 * (v_pad[:, 1:-1] + v_pad[:, 2:])
          * wm_pad[:, 1:-1] * wm_pad[:, 2:]
          * 0.5 * (p.cos_lat + jnp.roll(p.cos_lat, -1))[None, :, None])
    Fy = Fy.at[:, -1].set(0.0)                       # north truncation wall
    div_y = (Fy - jnp.roll(Fy, 1, axis=1)) * p.inv_dy / p.cos_lat[None, :, None]
    return div_x + div_y


def _layer_face_transports(velocity_x, velocity_y, params):
    """Static nodal volume flux per face width; y includes cos(face latitude)."""
    wet = params.wet_mask_z
    thickness_x = params.dz_node * wet * jnp.roll(wet, -1, axis=0)
    thickness_y = params.dz_node * wet * jnp.roll(wet, -1, axis=1)
    cosine_face = 0.5 * (params.cos_lat + jnp.roll(params.cos_lat, -1))
    flux_x = 0.5 * (velocity_x + jnp.roll(velocity_x, -1, axis=0)) * thickness_x
    flux_y = 0.5 * (velocity_y + jnp.roll(velocity_y, -1, axis=1)) * thickness_y * cosine_face[None, :, None]
    return flux_x, flux_y.at[:, -1].set(0.)


def _face_transport_divergence(flux_x, flux_y, params):
    """Divergence of layer or column face transports, with closed y walls."""
    incoming_y = jnp.roll(flux_y, 1, axis=1).at[:, 0].set(0.)
    inverse_dx = params.inv_dx[..., :1] if flux_x.ndim == 3 else params.inv_dx[..., 0]
    cosine = params.cos_lat[None, :, None] if flux_y.ndim == 3 else params.cos_lat[None, :]
    return ((flux_x - jnp.roll(flux_x, 1, axis=0)) * inverse_dx
            + (flux_y - incoming_y) * params.inv_dy / cosine)


def _match_layer_face_transports(velocity_x, velocity_y, column_transport, params):
    """Match the fast-mode time mean at every open face, retaining layer shear."""
    fluxes = _layer_face_transports(velocity_x, velocity_y, params)
    corrected = []
    for axis, flux, target in zip((0, 1), fluxes, column_transport, strict=True):
        thickness = params.dz_node * params.wet_mask_z * jnp.roll(params.wet_mask_z, -1, axis=axis)
        if axis == 1:
            thickness = thickness.at[:, -1].set(0.)
        depth = jnp.sum(thickness, axis=-1, keepdims=True)
        weights = thickness / jnp.where(depth > 0., depth, 1.)
        corrected.append(flux + weights * (target - jnp.sum(flux, axis=-1))[..., None])
    return tuple(corrected)


def _vertical_transport_iface(u, v, p, face_transport=None):
    """Interface volume transports Fz (nx, ny, nz+1), DOWNWARD-positive.

    The EXACT discrete inverse of _divergence_h, cumulated from the seafloor up:
    Fz[nz] = 0 and (Fz[k+1] - Fz[k])/dz_node[k] = -div_h[k], weighted by the
    per-node thickness dz_node (NOT dz_3d, which is interface-centred, length nz-1).
    Feeding this to the donor-cell vertical flux makes div_x + div_y + div_z cancel
    POINTWISE for a uniform tracer; a node-based w leaves a T-proportional residual
    no CFL limit removes. Fz[..., 0] is the column-integrated horizontal divergence,
    the rigid-lid leak the barotropic subcycle absorbs via Fz_top = Fz[0]*T[0]. (D7)
    """
    if face_transport is not None:
        integrand = _face_transport_divergence(*face_transport, p)
        accumulated = jnp.cumsum(integrand[..., ::-1], axis=-1)[..., ::-1]
        return jnp.concatenate([accumulated, jnp.zeros_like(accumulated[..., :1])], axis=-1)
    div_h = _divergence_h(u, v, p)
    # div_h is per-LAYER (nz entries) — weight by the per-node cell thickness
    # dz_node, NOT dz_3d (interface-centred, length nz-1).
    integrand = div_h * p.dz_node                    # (nx, ny, nz)
    # cumsum from the bottom upward: Fz_int[k] = sum_{m=k..nz-1} integrand[m]
    Fz_int = jnp.cumsum(integrand[..., ::-1], axis=-1)[..., ::-1]
    return jnp.concatenate([Fz_int, jnp.zeros_like(Fz_int[..., :1])], axis=-1)


# ── Vertical operators ─────────────────────────────────────────────

def _fill_ghost_bottom(u, p):
    """Seafloor no-flux fill: extend each column's bottom wet value downward.

    Ghost layers below the seafloor hold the T_ref/S_ref sentinel and never evolve,
    so any vertical stencil reaching them mixes that reservoir into the bottom wet
    layer. Replicating the bottom wet value is the standard MOM/NEMO no-flux
    treatment. Applied by every vertical operator. (D8)
    """
    wet = p.wet_mask_z
    idx = (wet > 0.5) * jnp.arange(u.shape[-1])
    kbot = jnp.max(idx, axis=-1)[..., None]        # (nx, ny, 1)
    kk = jnp.arange(u.shape[-1])[None, None, :]
    u_bot = jnp.take_along_axis(u, kbot, axis=-1)  # bottom wet value
    return jnp.where(kk >= kbot, u_bot, u)


def _d_dz(u, p):
    """Vertical first derivative, non-uniform grid."""
    du_interior = (u[..., 2:] - u[..., :-2]) / p.dz_denom_interior
    du_top = (u[..., 1:2] - u[..., 0:1]) / p.dz_bnd_top
    du_bot = (u[..., -1:] - u[..., -2:-1]) / p.dz_bnd_bot
    return jnp.concatenate([du_top, du_interior, du_bot], axis=-1)


def _d2_dz2(u, p):
    """Vertical second derivative, non-uniform grid.

    Top and bottom nodes use the zero-flux (ghost-point) form 2*(C1 - C0)/h0^2: the
    mirror ghost Cg = C1 makes the centered curvature diffusive at the boundary,
    where the plain centered-curvature form was anti-diffusive and drove the ITCZ
    salinity runaway. The field is ghost-filled first (_fill_ghost_bottom), so a
    seafloor that is not at the last grid level is a no-flux boundary too: without
    the fill the bottom wet node reads the T_ref/S_ref sentinel below the floor, a
    spurious seafloor flux up to 0.05 K/day. (D20)
    """
    u = _fill_ghost_bottom(u, p)
    d2u_interior = (
        u[..., 2:] * p.d2z_hm + u[..., :-2] * p.d2z_hp
        - u[..., 1:-1] * (p.d2z_hm + p.d2z_hp)
    ) / p.d2z_denom
    d2u_top = 2.0 * (u[..., 1:2] - u[..., 0:1]) / (p.d2z_h0_top ** 2)
    d2u_bot = 2.0 * (u[..., -2:-1] - u[..., -1:]) / (p.d2z_h0_bot ** 2)
    return jnp.concatenate([d2u_top, d2u_interior, d2u_bot], axis=-1)


def _iface_flux_divergence(Fz_i, p):
    """Divergence of an interface flux: tend[k] = (F[k-1/2] - F[k+1/2])/dz_node[k].

    Fz_i holds one flux per INTERFACE between nodes k and k+1 (length nz-1);
    the material top and bottom carry F = 0, which is the zero-flux BC. The
    column sum of tend*dz_node telescopes to F[bot]-F[top] = 0 exactly, for
    any flux and any mask (D9, D10, D18).
    """
    up = jnp.concatenate([jnp.zeros_like(Fz_i[..., :1]), Fz_i], axis=-1)
    dn = jnp.concatenate([Fz_i, jnp.zeros_like(Fz_i[..., :1])], axis=-1)
    return (up - dn) / p.dz_node


def _d2_dz2_flux(tracer, kappa, p):
    """Vertical diffusion in INTERFACE-flux form (exactly column-conservative).

    F[k+1/2] = -kappa*(C[k+1]-C[k])/dz_iface[k], zero at the material top and
    bottom; the column sum of tend*dz_node telescopes to F[bot]-F[top] = 0 for any
    kappa and any field. The node form does not telescope on a non-uniform grid, and
    vertical diffusion can only redistribute within a column. (D9)
    """
    # kappa may be scalar or a (nx, ny, nz) coastal-enhanced field; only the
    # scalar-zero shortcut can be branched on at trace time.
    if getattr(kappa, "ndim", 0) == 0 and kappa == 0.0:
        return jnp.zeros_like(tracer)
    tracer = _fill_ghost_bottom(tracer, p)
    Cm = tracer[..., :-1]
    Cp = tracer[..., 1:]
    Fz_i = -kappa * (Cp - Cm) / p.dz_iface                # (nx, ny, nz-1)
    wet_iface_f = (p.wet_mask_z[..., :-1] > 0.5) & (p.wet_mask_z[..., 1:] > 0.5)
    Fz_i = jnp.where(wet_iface_f, Fz_i, 0.0)
    return _iface_flux_divergence(Fz_i, p) * p.wet_mask_z


def _conv_flux_tendency(tracer, conv_mask_3d, kappa, p, iface_gate=None):
    """Convective mixing in INTERFACE-flux form (exactly column-conservative).

    Reuses the GM/Redi interface discretization, so the column sum telescopes to
    F[bot]-F[top] = 0 for ANY mask and the operator is negative-semidefinite. The
    node form (kappa_conv*mask*_d2_dz2) does not telescope, so every convective
    episode created net heat, polar-concentrated. (D10)

    Args: tracer (nx, ny, nz); conv_mask_3d (nx, ny, 1) column gate; kappa [m^2/s]
    (p.kappa_conv); optional iface_gate (nx, ny, nz-1) PER-INTERFACE gate, which
    makes mixing need both gates (localized convection).
    """
    if kappa <= 0.0:
        return jnp.zeros_like(tracer)
    # Seafloor no-flux fill (same T_ref=15 ghost guard as the Redi closure).
    tracer = _fill_ghost_bottom(tracer, p)
    Cm = tracer[..., :-1]                        # upper node value
    Cp = tracer[..., 1:]                         # lower node value
    Fz_i = -kappa * (Cp - Cm) / p.dz_iface       # (nx, ny, nz-1)
    # Flux only across wet interfaces AND only in convecting columns.
    wet_iface_f = (p.wet_mask_z[..., :-1] > 0.5) & (p.wet_mask_z[..., 1:] > 0.5)
    gate = (jnp.asarray(conv_mask_3d)[..., :1] > 0.5)   # force (nx, ny, 1)
    allow = gate & wet_iface_f
    if iface_gate is not None:
        allow = allow & (jnp.asarray(iface_gate) > 0.5)
    Fz_i = jnp.where(allow, Fz_i, 0.0)
    # Padded flux array carries the zero-flux BC at surface/seafloor;
    # tend[k] = (F[k-1/2] - F[k+1/2]) / dz_node[k] (see _redi_skew_flux_tendency).
    return _iface_flux_divergence(Fz_i, p) * p.wet_mask_z


def _effective_kappa_v(p):
    """Return the effective vertical diffusivity including the coastal band."""
    return p.kappa_v + p.coastal_kappa_v_2d[:, :, None]


def _vertical_diffusion(tracer, kappa, p):
    """Vertical diffusion of one tracer.

    `conservative_kv` selects the exactly-column-conservative interface-flux
    form; the default keeps the legacy node form bit-exact (D9).
    """
    if p.conservative_kv:
        return _d2_dz2_flux(tracer, kappa, p)
    return kappa * _d2_dz2(tracer, p)


def _vertical_momentum_diffusion(velocity, p):
    """Use the reference-cell flux balance for the nodal geometry candidate."""
    if p.column_geometry == 'nodal_dual_v1':
        return _d2_dz2_flux(velocity, p.nu_v, p)
    return p.nu_v * _d2_dz2(velocity, p)


def _convective_mask(state, p):
    """Convection gates from the density profile: (column, per-interface).

    An interface is unstable when the layer above is denser than the layer
    below AND both layers are wet. The both-wet guard is essential: the
    bottommost wet layer is denser than the masked ghost below it, so without
    it every wet/ghost interface tests unstable and convects the whole column
    (D10).
    """
    rho_prime = _density_anomaly(state.T, state.S, p) * p.wet_mask_z
    wet_iface = (p.wet_mask_z[..., :-1] > 0.5) & (p.wet_mask_z[..., 1:] > 0.5)
    unstable_iface = (rho_prime[..., :-1] > rho_prime[..., 1:]) & wet_iface
    return jnp.any(unstable_iface, axis=-1, keepdims=True), unstable_iface


def _subcycle(fn, carry, n, p):
    """Run `n` fixed-operator subcycles of `fn`, returning (carry, mean term).

    `fn(carry) -> (carry, term)` where `term` is that substep's tendency; the
    returned term is the mean over the substeps (None when the caller only
    wants the carry). The operator and any mask stay frozen across the
    subcycles, so the substep dt is what brings each term back inside its
    explicit CFL; `use_scan` fuses the loop into one compiled body and is
    numerically identical to the unrolled Python loop (D10, D12).
    """
    if p.use_scan:
        carry, terms = jax.lax.scan(lambda c, _: fn(c), carry, None, length=n)
        return carry, jax.tree.map(lambda t: t.mean(axis=0), terms)
    total = None
    for _ in range(n):
        carry, term = fn(carry)
        if total is None:
            total = term
        else:
            total = jax.tree.map(lambda a, b: a + b, total, term)
    return carry, jax.tree.map(lambda t: t / n, total)


# ── FD solver parameters (full, with physics + forcing) ───────────
# Extends FDParams (metric+grid) with the physics constants and forcing
# fields needed for the time integration. Built by make_solver_global.

FDPhysParams = namedtuple('FDPhysParams', [
    # metric + grid (from FDParams)
    'dx_2d', 'dy', 'cos_lat', 'inv_dx', 'inv_dy', 'inv_dx2', 'inv_dy2',
    'f', 'wet_mask', 'wet_mask_z',
    'interior_mask_z',
    'dz_denom_interior', 'dz_bnd_top', 'dz_bnd_bot',
    'd2z_hm', 'd2z_hp', 'd2z_denom', 'd2z_h0_top', 'd2z_h0_bot',
    'dz_3d', 'dz_surface', 'dz_iface', 'dz_node', 'surface_mask', 'bottom_mask',
    'nx', 'ny', 'nz',
    # physics
    'nu_h', 'nu_v', 'kappa_h', 'kappa_v', 'kappa_conv',
    'nu_bi', 'kappa_bi',
    'T_ref', 'S_ref', 'r_bot', 'cd', 'bottom_friction',
    # forcing (2D physical-space; no FFT pre-compute in the FD solver)
    'tau_x_2d', 'tau_y_2d', 'Q_heat_2d',
    # free surface
    'H_sw', 'dz_norm', 'dt',
    # bulk air-sea heat flux
    'T_atm_3d', 'lambda_bulk',
    # Surface salinity restoring (Haney): relax SSS toward S_clim_surf with an
    # equivalent salt flux, dSdt += -restore_coef_S*(S_surf - S_ref_2d)*surface_mask.
    # (D24)
    'S_ref_2d', 'restore_coef_S',
    # Diagnostic coastal surface-temperature restoring (land-adjacent band)
    'coastal_restore_coef_2d',  # (nx, ny) 1/s restoring coefficient
    'coastal_restore_T_2d',     # (nx, ny) target SST climatology
    # Diagnostic coastal extra bulk heat exchange
    'coastal_bulk_lambda_2d',   # (nx, ny) extra W/m^2/K
    # Diagnostic coastal horizontal tracer diffusivity
    'coastal_kappa_h_2d',       # (nx, ny) extra m^2/s
    # Diagnostic coastal vertical tracer diffusivity
    'coastal_kappa_v_2d',       # (nx, ny) extra m^2/s
    # lateral sponge (polar-edge Rayleigh damping; absorbs the wind-driven
    # barotropic energy that Laplacian dissipation can't arrest within its
    # CFL cap, which otherwise piles up at the polar edge rows)
    'sponge_rate',       # (nx, ny, 1) damping rate [1/s]; 0 interior
    'sponge_rate_2d',    # (nx, ny) 2D damping for the free-surface step
    'T_clim_3d', 'S_clim_3d',   # (nx, ny, nz) climatology the sponge relaxes to
    'polar_cap_rows',    # int: poleward rows fully zonally averaged per step (metric singularity)
    'polar_cap_taper',   # int: extra rows over which the cap blend cos^2-tapers 1->0 (cap-edge cliff)
    'dealias_lon_mask',  # (nx, 1, 1) 2/3-rule FFT dealias mask for the periodic lon axis
    # Gent-McWilliams eddy closure (sub-grid baroclinic transport)
    'kappa_gm',          # m²/s GM eddy diffusivity (bolus transport); 0 = off
    'gm_slope_max',      # dimensionless isopycnal-slope limiter
    'kappa_redi',        # m²/s Redi isopycnal diffusivity (skew-flux); 0 = off
    # Semi-enclosed-sea SSH (eta) relaxation (Mediterranean artifact fix)
    'eta_relax_mask',    # (nx, ny) 1 inside the semi-enclosed sea, 0 elsewhere
    'eta_relax_rate',    # 1/s Rayleigh relaxation rate; 0 = off
    # Mode split: with mode_split=True the free surface runs in n_subcyc
    # barotropic forward-backward subcycles of dt_bt at the end of _step_impl,
    # with the density-PGF coupling held fixed over the baroclinic step
    # (MOM-style forcing lag). Lifts the external-gravity-wave CFL off the
    # baroclinic dt (dt=3600 s is then safe at 1 deg). (D12)
    'mode_split',        # bool: barotropic-subcycle free surface
    'dt_bt',             # s: ACTUAL subcycle dt (= dt / n_subcyc, exact fill)
    'n_subcyc',          # int: barotropic subcycles per baroclinic step
    'conv_nsub',         # int: convective-adjustment subcycles per bc step
    'adv_nsub',          # int: vertical-advection subcycles per bc step
    # nu_h subcycle count in the split L half-steps. None = legacy
    # (n_nu = n_subcyc); an int right-sizes it from the actual metric at the
    # same FTCS margin, i.e. ~4x fewer Laplacian pairs per half-step (D12).
    'nu_nsub',
    # True: run the barotropic subcycle as a jax.lax.scan (fused device loop,
    # body compiled once) instead of an unrolled Python loop. Numerically
    # identical; cuts XLA graph size and host launch overhead.
    'use_scan',
    # True: freeze the tracer stage-2 advection velocity at the old (u, v)
    # instead of the raw forward-Euler predictor u_pred. Makes the tracer RK2
    # consistent with the momentum RK2 and removes the column heat leak. (D13)
    'freeze_adv_vel',
    # True: use the exactly-column-conservative interface-flux form of vertical
    # diffusion (_d2_dz2_flux) instead of kappa_v*_d2_dz2. Default False =
    # legacy bit-exact traces. (D9)
    'conservative_kv',
    # True: project the stage-2 tracer advection velocity so its column
    # divergence matches the final (subcycle-projected) u -- the continuity the
    # Fz_top = Fz[0]*T[0] closure needs to conserve column heat. Default False =
    # legacy bit-exact traces. (D13)
    'project_adv_vel',
    # True: gate the convective adjustment PER-INTERFACE (mix only across
    # interfaces that are actually unstable) instead of the column-wide mask,
    # which mixes the whole column whenever any interface is unstable. Default
    # False = legacy bit-exact traces. (D10)
    'localize_conv',
    # True: first-order donor-cell face values for the HORIZONTAL tracer fluxes
    # (the vertical tracer flux is already donor-cell), making the full 3D
    # tracer transport conservative and monotone for a divergence-free velocity.
    # Default False = historical centered path. (D16)
    'monotone_adv',
    # Experimental TVD/MUSCL flux limiter for horizontal tracer transport.
    # It reconstructs the face value from the two donor cells with a minmod
    # slope and chooses the state consistent with the face velocity. This is a
    # first local bounded-flux step toward a full Zalesak FCT scheme.
    'fct_adv',
    # Optional mixed-layer heat capacity: when set, surface heat flux is
    # distributed over this depth instead of the surface grid-cell thickness.
    # 0/None keeps the legacy bit-exact surface-node treatment.
    'mixed_layer_depth_m',
    # Optional 2D mask (1=apply mixed-layer depth, 0=legacy surface-node)
    'mixed_layer_mask_2d',
    # Optional 2D mixed-layer depth [m].  When present it overrides the scalar
    # mixed_layer_depth_m cell-by-cell while still respecting mixed_layer_mask_2d.
    'mixed_layer_depth_2d',
    # Minimal thermodynamic sea-ice closure: a constant brine-rejection salt
    # flux (psu/s) applied only where the surface temperature is at or below
    # the freezing point. 0 = off.
    'ice_freeze_temp_c',
    'ice_salt_flux',
    # Minimal dynamic ice: stateful thickness with latent-heat growth/melt,
    # conductivity insulation, and brine-rejection salinity flux.
    'dynamic_ice',
    'ice_insulation_scale_m',
    'projection_niter',
    'projection_rtol',
    'projection_preconditioner',
    'projection_inv_diagonal',
    'projection_niter_source',
    'projection_max_refinements',
    'column_geometry',
    'match_barotropic_transport',
    'process_time_scheme',
])

# Keyword-constructed callers that predate nu_nsub/use_scan/freeze_adv_vel/
# conservative_kv/project_adv_vel/localize_conv/monotone_adv get the legacy
# behavior instead of a TypeError.
FDPhysParams.__new__.__defaults__ = (None, False, False, False, False, False, False, False, None, None, None, -1.8, 0.0, False, 1.0, 150, None, 'none', None, 'default', 2, 'legacy', False, 'legacy')


# ── Equation of state ─────────────────────────────────────────────────

def _density_anomaly(T, S, p):
    """rho' = rho - rho_0 from the linear equation of state.

    The retired regional solver carried a UNESCO 1980 nonlinear branch
    behind an ``eos_type`` switch. It was never ported here -- the branch
    was a no-op stub returning this same expression -- so the switch is
    gone and the EOS is unambiguously linear. The convective-adjustment
    and buoyancy diagnostics are calibrated against this form.
    """
    return RHO_0 * (-ALPHA_T * (T - p.T_ref) + BETA_S * (S - p.S_ref))


def _compute_hydrostatic_pressure(state, p):
    """Full hydrostatic pressure via cumulative trapezoidal integration.

    The density anomaly is masked by wet_mask_z (zeroed below the seafloor) BEFORE
    integration, so ghost layers contribute dp = 0 and the pressure stays constant
    beneath the bottom: no spurious horizontal gradient from columns of different
    ghost-water length. (D11)
    """
    rho_prime = _density_anomaly(state.T, state.S, p)
    rho_prime = rho_prime * p.wet_mask_z          # zero out ghost water
    rho_avg = 0.5 * (rho_prime[..., :-1] + rho_prime[..., 1:])
    dp = G_EARTH * rho_avg * p.dz_3d
    p_bc = jnp.zeros_like(state.T)
    p_bc = p_bc.at[..., 1:].set(jnp.cumsum(dp, axis=-1))
    p_bt = RHO_0 * G_EARTH * state.eta[:, :, None]
    return p_bt + p_bc


def _compute_pressure_gradient(state, p):
    """Horizontal pressure gradient force per unit mass (FD).

    Uses _gradient_conservative_3d (masked, cos(lat)-weighted adjoint), NOT the
    bare centered _d_dx/_d_dy, so the 3D PGF does not reach into land zeros at
    coastlines (D4/D11 in docs/decisions.md).
    """
    pressure = _compute_hydrostatic_pressure(state, p)
    pgf_x, pgf_y = _gradient_conservative_3d(pressure, p)
    return -pgf_x / RHO_0, -pgf_y / RHO_0


def _barotropic_velocity(u, v, p):
    """Depth-averaged (barotropic) horizontal velocity."""
    if p.column_geometry == 'nodal_dual_v1':
        return jnp.sum(u * p.dz_norm, axis=-1), jnp.sum(v * p.dz_norm, axis=-1)
    u_avg = 0.5 * (u[..., :-1] + u[..., 1:])
    v_avg = 0.5 * (v[..., :-1] + v[..., 1:])
    ubt = jnp.sum(u_avg * p.dz_norm, axis=-1)
    vbt = jnp.sum(v_avg * p.dz_norm, axis=-1)
    return ubt, vbt


def _reference_depth_divergence(ubt, vbt, p):
    """Divergence of common-wet nodal face transport, not H_cell * div(u)."""
    return _column_divergence(ubt[..., None], vbt[..., None], p)


def _reference_depth_gradient(eta, p):
    """Negative adjoint of depth divergence in the A * H_ref velocity norm."""
    gradient_x, gradient_y = _gradient_conservative_3d(eta[..., None], p)
    return (jnp.sum(gradient_x * p.dz_norm, axis=-1),
            jnp.sum(gradient_y * p.dz_norm, axis=-1))


def _compute_bt_rho_pgf(state, p):
    """Barotropic (depth-averaged) PGF from density anomalies (FD).

    Depth average over the WET column of the same face-gated 3D baroclinic PGF the
    3D momentum feels (_gradient_conservative_3d), normalized by H_sw to match
    ubt = transport / H_sw in _barotropic_velocity:

        F_rho = (1/H_sw) * SUM_k pgf3d_layer_k * dz_k * wet_iface_k

    No ghost water and no below-bottom constant enters, so the spurious isobath
    forcing of the old trapezoidal-over-all-layers form is gone; what remains is
    the physical wet-column (JEBAR-type) coupling. (D26)
    """
    if p.column_geometry == 'nodal_dual_v1':
        acceleration_x, acceleration_y = _compute_pressure_gradient(state._replace(eta=jnp.zeros_like(state.eta)), p)
        return (jnp.sum(acceleration_x * p.dz_norm, axis=-1),
                jnp.sum(acceleration_y * p.dz_norm, axis=-1))
    rho_prime = _density_anomaly(state.T, state.S, p)
    rho_prime = rho_prime * p.wet_mask_z          # zero out ghost water
    rho_avg = 0.5 * (rho_prime[..., :-1] + rho_prime[..., 1:])
    dp = G_EARTH * rho_avg * p.dz_3d
    p_bc = jnp.zeros_like(state.T)
    p_bc = p_bc.at[..., 1:].set(jnp.cumsum(dp, axis=-1))
    # Layer-centered, face-gated 3D PGF, transport-weighted over the wet column.
    gx3, gy3 = _gradient_conservative_3d(p_bc, p)
    pgf_x_lay = 0.5 * (gx3[..., :-1] + gx3[..., 1:])
    pgf_y_lay = 0.5 * (gy3[..., :-1] + gy3[..., 1:])
    wet_iface = p.wet_mask_z[..., :-1] * p.wet_mask_z[..., 1:]
    fx = -jnp.sum(pgf_x_lay * wet_iface * p.dz_3d, axis=-1) / (RHO_0 * p.H_sw)
    fy = -jnp.sum(pgf_y_lay * wet_iface * p.dz_3d, axis=-1) / (RHO_0 * p.H_sw)
    return fx, fy


def _dealias_h_fd(field, p):
    """2-dx grid-scale filter for pointwise-nonlinear products (FD analogue of the
    spectral 2/3-rule dealias).

    Lon is periodic and uniformly spaced, so a 2/3 FFT rule is exact there. Lat is
    closed and non-uniform, so an FFT would mangle the field and a 5-pt binomial
    [1,4,6,4,1]/16 substitutes: 0 at the Nyquist wavenumber, ~0.92 at 8-dx. Edge
    padding mirrors the wall ghost cell, as in _d_dy/_laplacian_h.

    Applied at exactly THREE call sites, all pointwise nonlinear products: adv_u,
    adv_v, and the GM/Redi skew-flux tendency. Deliberately NOT applied to the
    diagnosed w or to the flux-form tracer advection, whose exact discrete
    telescoping it would destroy (D7, D15). (D25)
    """
    # 2/3 FFT dealias in lon (axis 0, periodic + uniform -> exact).
    f_hat = jnp.fft.fft(field, axis=0)
    f_hat = f_hat * p.dealias_lon_mask
    f_lon = jnp.real(jnp.fft.ifft(f_hat, axis=0))
    # 5-pt binomial low-pass [1,4,6,4,1]/16 in lat (axis 1, edge-padded wall).
    fp = jnp.pad(f_lon, ((0, 0), (2, 2), (0, 0)), mode='edge')
    f_sm = (fp[:, :-4] + 4.0 * fp[:, 1:-3] + 6.0 * fp[:, 2:-2]
            + 4.0 * fp[:, 3:-1] + fp[:, 4:]) / 16.0
    return f_sm


def _compute_vertical_velocity(state, p):
    """Diagnose w from horizontal continuity. w=0 at bottom. Masked on land.

    w = -Fz, the exact interface transports of _vertical_transport_iface sign-flipped
    to the w>0 = DOWNWARD convention. A trapezoid node-w built from div_avg + cumsum
    could not invert the advection's discrete face-flux divergence pointwise, leaving
    a T-proportional source (an exponential pump); the exact inverse closes
    div_x + div_y + div_z to machine precision for a uniform tracer. No dealiasing:
    dealiasing w would re-open the non-closure the exact inverse just fixed.
    (D6, D7)
    """
    Fz = _vertical_transport_iface(state.u, state.v, p)
    w = -Fz[..., :-1] * p.wet_mask_z
    return w


# ── Tendencies (FD, with land masking) ─────────────────────────────

def _gradient_face_gated_3d(field, p):
    """Face-gated horizontal gradient for advection: d/dx, d/dy with each face
    difference zeroed at every wet/ghost (or wet/dry) interface.

    The bare centered _d_dx/_d_dy differences across mask boundaries: ghost nodes
    hold the T_ref sentinel (+15 C) against ~+1 C deep water, so the centered
    stencil reads dT ~ 14 K across every wet/ghost face and a deep coastal current
    feels advection from a temperature that does not exist. Zeroing the face
    difference matches the already-gated mass flux and _gradient_conservative_3d;
    where the flow is parallel to the boundary the cross-boundary flux is zero
    anyway, so only the unphysical part is removed. (D14)

    Returns (dF/dx, dF/dy) on wet nodes; ghost nodes keep whatever the caller masks.
    """
    wm = p.wet_mask_z                                   # (nx, ny, nz)
    # Zonal (axis 0, periodic): face (i+1/2) open iff cells i, i+1 both wet.
    open_xp = wm * jnp.roll(wm, -1, axis=0)
    open_xm = wm * jnp.roll(wm, 1, axis=0)
    d_fp = (jnp.roll(field, -1, axis=0) - field) * open_xp
    d_fm = (field - jnp.roll(field, 1, axis=0)) * open_xm
    grad_x = p.inv_dx[..., 0:1] * 0.5 * (d_fp + d_fm)
    # Meridional (axis 1, closed N/S walls): mirror-ghost edge padding as in
    # _d_dy (zero normal gradient at the wall) PLUS the wet/wet face gate.
    pad = [(0, 0), (1, 1), (0, 0)]
    f_pad = jnp.pad(field, pad, mode='edge')
    wm_pad = jnp.pad(wm, pad, mode='edge')
    open_yp = wm_pad[:, 2:] * wm_pad[:, 1:-1]           # face (j+1/2)
    open_ym = wm_pad[:, 1:-1] * wm_pad[:, :-2]          # face (j-1/2)
    d_yp = (f_pad[:, 2:] - f_pad[:, 1:-1]) * open_yp
    d_ym = (f_pad[:, 1:-1] - f_pad[:, :-2]) * open_ym
    grad_y = p.inv_dy * 0.5 * (d_yp + d_ym)
    return grad_x, grad_y


def _advection_flux_form(u, v, w, p):
    """3D advective-form momentum advection (FD, land-masked, dealiased).

    Advective form avoids the spurious u*div_h source; momentum keeps it while
    tracers use the flux form (D15). Horizontal gradients use
    _gradient_face_gated_3d (D14); the vertical gradient stays the bare _d_dz,
    because w is masked to zero in ghost layers and the terms are multiplied by
    wet_mask_z, so a wet/ghost vertical face carries no flux whatever dT/dz reads.
    The summed nonlinear tendency is de-aliased once by _dealias_h_fd, then
    land-masked. (D25)
    """
    du_dx, du_dy = _gradient_face_gated_3d(u, p)
    dv_dx, dv_dy = _gradient_face_gated_3d(v, p)
    du_dz = _d_dz(_fill_ghost_bottom(u, p), p)
    dv_dz = _d_dz(_fill_ghost_bottom(v, p), p)
    adv_u = -(u * du_dx + v * du_dy + w * du_dz)
    adv_v = -(u * dv_dx + v * dv_dy + w * dv_dz)
    adv_u = _dealias_h_fd(adv_u, p)
    adv_v = _dealias_h_fd(adv_v, p)
    return adv_u * p.wet_mask_z, adv_v * p.wet_mask_z


def _limited_tracer_slope(tracer, wet, axis):
    """Minmod reconstruction using only wet neighbours and physical boundaries."""
    if axis == 1:
        padding = [(0, 0), (1, 1), (0, 0)]
        padded = jnp.pad(tracer, padding, mode='edge')
        padded_wet = jnp.pad(wet, padding, mode='edge')
        previous, following = padded[:, :-2], padded[:, 2:]
        previous_wet, following_wet = padded_wet[:, :-2], padded_wet[:, 2:]
    else:
        previous, following = jnp.roll(tracer, 1, axis=axis), jnp.roll(tracer, -1, axis=axis)
        previous_wet, following_wet = jnp.roll(wet, 1, axis=axis), jnp.roll(wet, -1, axis=axis)
    left_delta, right_delta = tracer - previous, following - tracer
    signs = jnp.sign(left_delta) + jnp.sign(right_delta)
    return (0.5 * signs * jnp.minimum(jnp.abs(left_delta), jnp.abs(right_delta))
            * wet * previous_wet * following_wet)


def _advection_scalar(T, u, v, Fz_in, p, return_boundary=False, face_transport=None):
    """3D FLUX-FORM scalar advection (FD, land-masked, NOT dealiased).

    Flux form (-div(uT)) rather than advective form: the two differ by +T*div(u), a
    REAL anti-diffusion whose positive eigenvalue dt*delta grows the field
    exponentially at any nonzero discrete divergence, and subcycling only slows it
    without removing it. Flux form is exactly conservative and matches the
    continuity-consistent tracer equation the free-surface subcycle solves; momentum
    keeps the advective form. (D15)

    No dealiasing here, unlike _advection_flux_form: the flux form's value is that
    its column divergence telescopes exactly into the boundary fluxes, which is what
    closes the column heat budget, and _dealias_h_fd would break it. Horizontal
    fluxes are face-gated at mask boundaries (D14); the vertical flux is donor-cell
    on Fz_in from _vertical_transport_iface, the exact discrete inverse of the
    horizontal divergence, so the column transport is exactly conservative and
    monotone. (D7, D16)

    ``fct_adv`` is a compact TVD/MUSCL flux limiter: it reconstructs the face
    state from the two donor cells with a minmod slope and chooses the state
    consistent with the face velocity. It is not yet a full Zalesak multidimensional
    FCT limiter, but it is conservative and removes the centered scheme's local
    overshoot at a sharp front.
    """
    wm = p.wet_mask_z
    # ── Zonal flux at face (i+1/2), periodic in x ──
    # Fx = u_face * T_face (centered unless monotone_adv), face-gated on both
    # cells wet; +x-directed (u>0 carries T eastward).
    ux_face = 0.5 * (u + jnp.roll(u, -1, axis=0))
    if face_transport is not None:
        ux_face = face_transport[0] / p.dz_node
    if p.fct_adv:
        # TVD/MUSCL flux limiter: reconstruct from the left and right cells with a
        # minmod slope, then choose the state consistent with the face velocity.
        # This is a compact local limiter, not a full Zalesak 3D FCT, but it is
        # flux-form, conservative, and removes the centered scheme's overshoot at
        # a sharp front while remaining second-order in smooth regions.
        slope_x = _limited_tracer_slope(T, wm, axis=0)
        Tx_face = jnp.where(
            ux_face >= 0.0,
            T + 0.5 * slope_x,
            jnp.roll(T, -1, axis=0) - 0.5 * jnp.roll(slope_x, -1, axis=0))
    elif p.monotone_adv:
        # Donor-cell face value: the upstream cell (positive ux takes T[i],
        # negative ux takes T[i+1]). Lower order than centered flux, but
        # conservative and positivity-preserving for Courant <= 1.
        Tx_face = jnp.where(ux_face >= 0.0, T, jnp.roll(T, -1, axis=0))
    else:
        Tx_face = 0.5 * (T + jnp.roll(T, -1, axis=0))
    gate_x = wm * jnp.roll(wm, -1, axis=0)
    Fx = ux_face * Tx_face * gate_x                    # (nx, ny, nz)
    # ── Meridional flux at face (j+1/2), closed N/S walls ──
    # The v*T flux carries the same cos(face_lat) factor as the mass flux in
    # _divergence_conservative, and the cell divergence divides by cos(cell).
    pad = [(0, 0), (1, 1), (0, 0)]
    T_pad = jnp.pad(T, pad, mode='edge')
    v_pad = jnp.pad(v, pad, mode='edge')
    wm_pad = jnp.pad(wm, pad, mode='edge')
    vy_face = 0.5 * (v_pad[:, 1:-1] + v_pad[:, 2:])
    if face_transport is not None:
        cosine_face = 0.5 * (p.cos_lat + jnp.roll(p.cos_lat, -1))
        vy_face = face_transport[1] / (p.dz_node * cosine_face[None, :, None])
    if p.fct_adv:
        # Same TVD/MUSCL limiter as x, but on the padded row. The last face is
        # closed explicitly below, so the edge-padded value cannot enter the
        # budget.
        slope_y = _limited_tracer_slope(T, wm, axis=1)
        slope_y_pad = jnp.pad(slope_y, pad, mode='edge')
        Ty_face = jnp.where(
            vy_face >= 0.0,
            T_pad[:, 1:-1] + 0.5 * slope_y_pad[:, 1:-1],
            T_pad[:, 2:] - 0.5 * slope_y_pad[:, 2:])
    elif p.monotone_adv:
        # Same donor-cell choice as x; the last face is closed below, so the
        # edge-padded value cannot enter.
        Ty_face = jnp.where(vy_face >= 0.0, T_pad[:, 1:-1], T_pad[:, 2:])
    else:
        Ty_face = 0.5 * (T_pad[:, 1:-1] + T_pad[:, 2:])
    gate_y = wm_pad[:, 1:-1] * wm_pad[:, 2:]
    cos_face = 0.5 * (p.cos_lat + jnp.roll(p.cos_lat, -1))
    Fy = vy_face * Ty_face * gate_y * cos_face[None, :, None]   # (nx, ny, nz)
    # The edge pad replicates wm at the last row, so Fy[:, -1] would carry a
    # flux through the closed wall (v is masked to 0 there, but vy_face reads
    # the interior neighbour). Close it explicitly, as
    # _divergence_conservative does.
    Fy = Fy.at[:, -1].set(0.0)
    # ── Vertical flux at interface (k+1/2): DONOR-CELL (upwind) ──
    # Fz is the DOWNWARD-positive interface transport (nx, ny, nz+1) from
    # _vertical_transport_iface; the donor is the SHALLOWER node when Fz > 0 and
    # the DEEPER node when Fz < 0. The face value must NOT be centered: centered
    # vertical transport is unconditionally unstable under RK2/forward-Euler,
    # and the centered value multiplies the full tracer content into the surface
    # node wherever Fz[0] != 0. Donor-cell is monotone and adds only
    # 0.5*|w|*dz of implicit diffusivity (below kappa_v). The bottom face reads
    # _fill_ghost_bottom(T) for the seafloor no-flux BC. (D16)
    T_deep = _fill_ghost_bottom(T, p)[..., 1:]        # node k (below iface k)
    T_shallow = T[..., :-1]                           # node k-1 (above iface k)
    wet_iface = wm[..., :-1] * wm[..., 1:]
    # Internal interfaces are Fz_in[..., 1:-1] (indices 1..nz-1, count nz-1):
    # index 0 is the surface (rigid-lid leak, handled below) and index nz is
    # the seafloor (zero transport).
    Fz_int = Fz_in[..., 1:-1] * jnp.where(Fz_in[..., 1:-1] > 0.0, T_shallow, T_deep) * wet_iface
    # Tendency is OUT-minus-IN: k increases DOWNWARD, so cell k's out-face is
    # its bottom (k+1/2) and its in-face the top (k-1/2). Swapping to (up - dn)
    # makes the donor-cell scheme ANTI-upwind. (D16)
    Fx_up = jnp.roll(Fx, 1, axis=0)                   # flux at face (i-1/2)
    Fy_up = jnp.roll(Fy, 1, axis=1)
    Fy_up = Fy_up.at[:, 0].set(0.0)                   # south wall closed
    # Top face: Fz[..., 0] is the column-integrated horizontal divergence, the
    # rigid-lid leak the barotropic subcycle absorbs as the eta tendency. The
    # MOM-style rigid-lid closure carries the surface cell's own value,
    # Fz_top = Fz[0]*T[0], so upwelling equilibrates the surface node at the
    # deep value instead of growing without bound. (D7, D16)
    Fz_top = Fz_in[:, :, :1] * T[..., :1]
    up = jnp.concatenate([Fz_top, Fz_int], axis=-1)
    dn = jnp.concatenate([Fz_int, jnp.zeros_like(Fz_int[..., :1])], axis=-1)
    div_x = (Fx - Fx_up) * p.inv_dx[..., 0:1]
    # Meridional: divide by cos(cell lat) like _divergence_conservative.
    div_y = (Fy - Fy_up) * p.inv_dy / p.cos_lat[None, :, None]
    div_z = (dn - up) / p.dz_node
    adv_T = -(div_x + div_y + div_z)
    tendency = adv_T * p.wet_mask_z
    if return_boundary:
        return tendency, Fz_top[..., 0]
    return tendency


def _compute_momentum_tendency(state, p):
    """du/dt, dv/dt for hydrostatic primitive equations (FD)."""
    w = _compute_vertical_velocity(state, p)
    adv_u, adv_v = _advection_flux_form(state.u, state.v, w, p)

    f_3d = p.f[:, :, None]
    cor_u = f_3d * state.v
    cor_v = -f_3d * state.u

    pgf_x, pgf_y = _compute_pressure_gradient(state, p)

    diff_h_u = p.nu_h * _laplacian_h(state.u, p)
    diff_h_v = p.nu_h * _laplacian_h(state.v, p)
    diff_v_u = _vertical_momentum_diffusion(state.u, p)
    diff_v_v = _vertical_momentum_diffusion(state.v, p)

    wind_factor = 1.0 / (RHO_0 * p.dz_surface)
    wind_u = p.tau_x_2d[:, :, None] * wind_factor * p.surface_mask
    wind_v = p.tau_y_2d[:, :, None] * wind_factor * p.surface_mask

    if p.bottom_friction == 'quadratic':
        speed = jnp.sqrt(state.u**2 + state.v**2)
        bot_u = -p.cd * speed * state.u * p.bottom_mask
        bot_v = -p.cd * speed * state.v * p.bottom_mask
    else:
        bot_u = -p.r_bot * state.u * p.bottom_mask
        bot_v = -p.r_bot * state.v * p.bottom_mask

    dudt = adv_u + cor_u + pgf_x + diff_h_u + diff_v_u + wind_u + bot_u
    dvdt = adv_v + cor_v + pgf_y + diff_h_v + diff_v_v + wind_v + bot_v
    # Land: hold velocity at rest (no tendency over land).
    dudt = dudt * p.wet_mask_z
    dvdt = dvdt * p.wet_mask_z
    return dudt, dvdt


# ── Gent-McWilliams sub-grid baroclinic closure ────────────────────
# Unresolved baroclinic eddies as an advective bolus transport that flattens
# isopycnal slopes, releasing baroclinic available potential energy. Required at
# coarse (1 deg) resolution where the Rossby radius (~30-50 km) is sub-grid.
# Tracer-only (not applied to momentum); gated by p.kappa_gm > 0. (D18)

# Vertical stratification floor [kg/m^4] guards ∂rho'/∂z against division
# blow-up in weakly-stratified / convective columns ( rho' ~ well-mixed ).
_GM_RHOZ_FLOOR = 1.0e-5

# Target value of dt*D_v*(1/h_k + 1/h_k1)/dz_iface (explicit-vertical-diffusion
# CFL number) for the Redi S^2 vertical skew term, per interface. See the
# comment at the cap site in _redi_skew_flux_tendency.
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


# ── Redi isopycnal mixing (dissipative counterpart to GM) ──────────
# Full isoneutral diffusion (z-up, matching _d_dz). The -k grad_h C part of
# -k grad_rho C is the background kappa_h lap the linear step already applies
# (and _compute_tracer_residual subtracts back), so the Griffies skew-flux
# RESIDUAL form keeps only the slope-driven remainder:
#     F^h_skew = -k S d_z C        F^z_skew = -k (S.grad_h C + |S|^2 d_z C)
# The vertical S.grad_h C cross-term is essential: the horizontal skew flux
# alone can STEEPEN a front. Tendency = -div(F_skew), dealiased and masked.
# (D18)

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


def _tracer_terms(state, p):
    """Diagnostic decomposition of dT/dt into physical terms.

    Returns the (6, nx, ny, nz) stack [adv, diff_h, diff_v, conv, gm, redi]
    of land-masked T-terms. Read-only diagnosis (blow-up attribution); NOT
    used in the time integration. Mirrors _compute_tracer_tendency.
    """
    Fz = _vertical_transport_iface(state.u, state.v, p)
    adv_T = _advection_scalar(state.T, state.u, state.v, Fz, p)
    diff_h_T = _horizontal_tracer_diffusion(state.T, p)
    diff_v_T = _vertical_diffusion(state.T, _effective_kappa_v(p), p)

    conv_mask_3d, unstable_iface = _convective_mask(state, p)
    conv_T = _conv_flux_tendency(state.T, conv_mask_3d, p.kappa_conv, p,
                                 unstable_iface if p.localize_conv else None)

    gm_T, _, redi_T, _ = _isopycnal_closure(state, p)
    terms = [adv_T, diff_h_T, diff_v_T, conv_T, gm_T, redi_T]
    return jnp.stack([t * p.wet_mask_z for t in terms], axis=0)


def _surface_heat_weights(p):
    """Per-node heat deposition; sum(weights * wet node thickness) is one."""
    mixed_depth = float(p.mixed_layer_depth_m or 0.)
    if mixed_depth <= 0. and p.mixed_layer_depth_2d is None:
        return p.surface_mask * p.wet_mask_z / p.dz_surface
    requested_depth = (p.mixed_layer_depth_2d if p.mixed_layer_depth_2d is not None
                       else jnp.full_like(p.wet_mask, mixed_depth))
    depth = jnp.where(p.mixed_layer_mask_2d > 0.5, requested_depth, p.dz_surface)
    layer_top = jnp.cumsum(p.dz_node, axis=-1) - p.dz_node
    overlap = jnp.clip(depth[:, :, None] - layer_top, 0., p.dz_node) * p.wet_mask_z
    wet_depth = jnp.maximum(jnp.sum(overlap, axis=-1, keepdims=True), 1e-12)
    return overlap / (wet_depth * p.dz_node)


def _compute_tracer_tendency(state, p, budget=None, face_transport=None, return_terms=False):
    """dT/dt, dS/dt (FD, land-masked). Includes bulk air-sea heat flux.

    Vertical advection is SUBCYCLED adv_nsub times when adv_nsub > 1 (frozen
    velocity and horizontal structure, fixed-operator subcycling like conv_nsub):
    the explicit advection CFL is dt*(|u|/dx + |v|/dy + w/dz) < ~1, dominated by
    w/dz in the thin surface layer. Only the ADVECTIVE term is subcycled --
    diffusion, surface fluxes, GM/Redi skew flux and convective adjustment are all
    inside their explicit bounds at dt=3600 and stay evaluated ONCE. (D16)
    """
    measure_terms = budget is not None or return_terms
    Fz = _vertical_transport_iface(state.u, state.v, p, face_transport=face_transport)
    if budget is not None and p.column_geometry == 'nodal_dual_v1':
        budget.tracer_transport(face_transport if face_transport is not None
                                else _layer_face_transports(state.u, state.v, p))
    n_a = int(p.adv_nsub)
    if n_a > 1:
        # The reported rate is the MEAN over the substeps, so the Strang
        # residual (which subtracts it and re-applies it over the full dt in
        # the N-step RK2) integrates the same subcycled operator over dt.
        dts = p.dt / n_a

        def _adv_sub(carry):
            tT, tS = carry
            if measure_terms:
                aT, top_T = _advection_scalar(tT, state.u, state.v, Fz, p, return_boundary=True, face_transport=face_transport)
                aS, top_S = _advection_scalar(tS, state.u, state.v, Fz, p, return_boundary=True, face_transport=face_transport)
                terms = (aT, aS, top_T, top_S)
            else:
                aT = _advection_scalar(tT, state.u, state.v, Fz, p, face_transport=face_transport)
                aS = _advection_scalar(tS, state.u, state.v, Fz, p, face_transport=face_transport)
                terms = (aT, aS)
            return (tT + aT * dts, tS + aS * dts), terms

        _, terms = _subcycle(_adv_sub, (state.T, state.S), n_a, p)
        adv_T, adv_S = terms[:2]
        if measure_terms:
            top_T, top_S = terms[2:]
    else:
        if measure_terms:
            adv_T, top_T = _advection_scalar(state.T, state.u, state.v, Fz, p, return_boundary=True, face_transport=face_transport)
            adv_S, top_S = _advection_scalar(state.S, state.u, state.v, Fz, p, return_boundary=True, face_transport=face_transport)
        else:
            adv_T = _advection_scalar(state.T, state.u, state.v, Fz, p, face_transport=face_transport)
            adv_S = _advection_scalar(state.S, state.u, state.v, Fz, p, face_transport=face_transport)

    diff_h_T = _horizontal_tracer_diffusion(state.T, p)
    diff_h_S = _horizontal_tracer_diffusion(state.S, p)
    diff_v_T = _vertical_diffusion(state.T, _effective_kappa_v(p), p)
    diff_v_S = _vertical_diffusion(state.S, _effective_kappa_v(p), p)

    # The legacy subcycle scales both kappa and time. The candidate retains the
    # physical kappa and divides time only; neither path proves RK order. (D10)
    conv_mask_3d, unstable_iface = _convective_mask(state, p)
    iface_gate = unstable_iface if p.localize_conv else None
    n_c = int(p.conv_nsub)
    if n_c > 1:
        kappa_c = p.kappa_conv if p.process_time_scheme != 'legacy' else p.kappa_conv / n_c
        h_c = p.dt / n_c

        def _conv_sub(carry):
            tT, tS = carry
            ttT = _conv_flux_tendency(tT, conv_mask_3d, kappa_c, p, iface_gate)
            ttS = _conv_flux_tendency(tS, conv_mask_3d, kappa_c, p, iface_gate)
            return (tT + ttT * h_c, tS + ttS * h_c), (ttT, ttS)

        _, (conv_T, conv_S) = _subcycle(_conv_sub, (state.T, state.S), n_c, p)
    else:
        conv_T = _conv_flux_tendency(state.T, conv_mask_3d, p.kappa_conv, p, iface_gate)
        conv_S = _conv_flux_tendency(state.S, conv_mask_3d, p.kappa_conv, p, iface_gate)

    heat_factor = _surface_heat_weights(p) / (RHO_0 * C_P)
    heat_T = p.Q_heat_2d[:, :, None] * heat_factor
    # Bulk air-sea heat flux (Haney/Barnier): genuine SST negative feedback.
    bulk_T = (p.lambda_bulk * (p.T_atm_3d - state.T[:, :, 0:1])
              * heat_factor)
    coastal_bulk_T = (p.coastal_bulk_lambda_2d[:, :, None]
                      * (p.T_atm_3d - state.T[:, :, 0:1])
                      * heat_factor)
    if getattr(p, 'dynamic_ice', False):
        heat_T = jnp.zeros_like(heat_T)
        bulk_T = jnp.zeros_like(bulk_T)
        coastal_bulk_T = jnp.zeros_like(coastal_bulk_T)

    # Surface salinity restoring (Haney): equivalent salt flux relaxing SSS
    # to climatology with timescale tau = 1/restore_coef_S. Same form as the
    # bulk heat flux — a surface tracer BC, not a body source (surface_mask).
    rest_S = (p.restore_coef_S * (p.S_ref_2d[:, :, None] - state.S[:, :, 0:1])
              * p.surface_mask)

    # Minimal thermodynamic sea-ice closure: add brine-rejection salt flux
    # where the live SST is at or below the freezing point.  This is opt-in
    # and does not alter the heat budget until enabled.
    ice_salt = (p.ice_salt_flux
                * (state.T[:, :, 0:1] <= p.ice_freeze_temp_c)
                * p.surface_mask)
    if getattr(p, 'dynamic_ice', False):
        ice_salt = jnp.zeros_like(ice_salt)

    # Diagnostic coastal surface-temperature restoring. Same Haney form as the
    # salinity restoring, but restricted to a land-adjacent mask and used only
    # to attribute how much of the SST error is local.
    rest_T = (p.coastal_restore_coef_2d[:, :, None]
              * (p.coastal_restore_T_2d[:, :, None] - state.T[:, :, 0:1])
              * p.surface_mask)

    gm_T, gm_S, redi_T, redi_S = _isopycnal_closure(state, p)

    dTdt = (adv_T + diff_h_T + diff_v_T + heat_T + bulk_T + coastal_bulk_T
            + conv_T + gm_T + redi_T + rest_T)
    dSdt = (adv_S + diff_h_S + diff_v_S + conv_S + gm_S + redi_S
            + rest_S + ice_salt)
    # Land: tracers held (no tendency over land).
    dTdt = dTdt * p.wet_mask_z
    dSdt = dSdt * p.wet_mask_z
    if budget is not None:
        budget.surface_sources((heat_T, bulk_T, coastal_bulk_T, rest_T, rest_S, ice_salt))
        budget.nonlinear_terms(((adv_T, adv_S), (conv_T, conv_S), (gm_T, gm_S), (redi_T, redi_S)))
        budget.advection_boundary_fluxes(top_T, top_S)
    if return_terms:
        processes = ((adv_T, adv_S), (conv_T, conv_S), (gm_T, gm_S), (redi_T, redi_S))
        return dTdt, dSdt, ((heat_T, bulk_T, coastal_bulk_T, rest_T, rest_S, ice_salt),
                            processes, top_T, top_S, jax.tree.map(jnp.abs, processes))
    return dTdt, dSdt


# ── Linear half-step (FD: explicit diffusion + exact Coriolis + free surface) ─

def _coriolis_rotation_2d(u, v, f, dt):
    """Exact Coriolis rotation on the full 2D f-field (not f-plane).

    Per-gridpoint rotation: u' = cos(f*dt)*u + sin(f*dt)*v, etc.
    f varies with latitude (2D), so this is exact everywhere, no beta-plane
    approximation.
    """
    angle = f[:, :, None] * dt     # broadcast 2D f to 3D velocity fields
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    u_new = cos_a * u + sin_a * v
    v_new = -sin_a * u + cos_a * v
    return u_new, v_new


def _refill_volume(eta_now, eta_before, area_cell, area_ocean, p):
    """Spread a decay-removed volume uniformly back over the wet domain.

    A Rayleigh decay changes global volume by sum(A*(eta_now - eta_before)).
    Adding that back as a uniform eta offset conserves total volume exactly,
    and a uniform offset has zero PGF so the dynamics are untouched. Used by
    the lateral sponge and the semi-enclosed-sea eta relaxation.
    """
    dV = jnp.sum(area_cell * (eta_now - eta_before))
    return eta_now + (-dV / area_ocean) * p.wet_mask


def _filter_barotropic_eta(eta, params, duration):
    """Retain the actual sponge/relaxation/cap changes, independently of transport."""
    area = params.dx_2d * params.dy
    ocean_area = jnp.maximum(jnp.sum(area * params.wet_mask), 1.0)
    decay = jnp.exp(-params.sponge_rate_2d * duration)
    updated = _refill_volume(eta * decay, eta, area, ocean_area, params)
    relaxation = jnp.exp(-params.eta_relax_rate * duration * params.eta_relax_mask)
    updated = _refill_volume(updated * relaxation, updated, area, ocean_area, params)
    return _apply_polar_cap(updated, params.wet_mask, params)


def _symmetric_free_surface_step(eta, velocity_x, velocity_y, params, duration,
                                 forcing_x=None, forcing_y=None, column_face_transport=None):
    """Drift-kick-drift with constrained midpoint rotation and actual half-step faces."""
    normal_mask = params.interior_mask_z[..., 0]
    velocity_x = velocity_x * params.wet_mask
    velocity_y = velocity_y * params.wet_mask * normal_mask
    if column_face_transport is None:
        layers = _layer_face_transports(velocity_x[..., None], velocity_y[..., None], params)
        first_faces = tuple(jnp.sum(flux, axis=-1) for flux in layers)
    else:
        first_faces = column_face_transport
    midpoint_eta = eta - (duration / 2.) * _face_transport_divergence(*first_faces, params)
    pressure_x, pressure_y = _reference_depth_gradient(midpoint_eta, params)
    source_x = params.tau_x_2d / (RHO_0 * params.H_sw)
    source_y = params.tau_y_2d / (RHO_0 * params.H_sw)
    if forcing_x is not None:
        source_x, source_y = source_x + forcing_x, source_y + forcing_y
    half_rotation = params.f * normal_mask * (duration / 2.)
    right_x = velocity_x + half_rotation * velocity_y + duration * (-G_EARTH * pressure_x + source_x)
    right_y = (velocity_y - half_rotation * velocity_x
               + duration * (-G_EARTH * pressure_y + source_y)) * normal_mask
    denominator = 1. + half_rotation * half_rotation
    next_x = ((right_x + half_rotation * right_y) / denominator) * params.wet_mask
    next_y = ((right_y - half_rotation * right_x) / denominator) * params.wet_mask * normal_mask
    delta_layers = _layer_face_transports((next_x - velocity_x)[..., None],
                                         (next_y - velocity_y)[..., None], params)
    second_faces = tuple(first + jnp.sum(delta, axis=-1)
                         for first, delta in zip(first_faces, delta_layers, strict=True))
    transported_eta = midpoint_eta - (duration / 2.) * _face_transport_divergence(*second_faces, params)
    mean_faces = tuple(0.5 * (first + second) for first, second in zip(first_faces, second_faces, strict=True))
    final_eta = _filter_barotropic_eta(transported_eta * params.wet_mask, params, duration)
    decay = jnp.exp(-params.sponge_rate_2d * duration)
    final_x = _apply_polar_cap(next_x * decay, params.wet_mask, params)
    final_y = _apply_polar_cap(next_y * decay, params.wet_mask, params) * normal_mask
    return (final_eta, final_x, final_y), mean_faces


def _free_surface_step_fd(eta, u, v, p, F_rho_x=None, F_rho_y=None, dt_half=None,
                          column_divergence_offset=None, column_face_transport=None):
    """Forward-backward (Sielecki) free-surface (shallow water) step on lat-lon FD.

        eta^{n+1} = eta^n - dt*H_sw*div_h(ubt^n)                   # eta from OLD u
        ubt^{n+1} = (ubt^n + dt*(-g*grad_h(eta^{n+1}) + F)) / (1 + r_bt*dt)

    with implicit linear barotropic bottom drag (unconditionally stable). The
    forward-forward pairing (both from the old state) is unconditionally unstable
    for free gravity waves -- |lambda| = sqrt(1+(dt*c*k)^2) > 1 for ALL k -- while
    forward-backward gives |lambda| = 1 under the external-wave CFL dt < dx/sqrt(gH).
    (D22)

    mode_split=True: called n_subcyc times per baroclinic step with dt_bt, one
    forward-backward pair per call; the 3D<->barotropic projection stays in the
    CALLER, so the subcycle sees only the 2D (eta, ubt, vbt) state. The subcycle
    carries NO nu_h: the baroclinic shear nu_h must damp is invisible to the
    depth-averaged BT state, so nu_h runs in the L half-steps on the full 3D field
    with its own subcycle. (D12)
    """
    if dt_half is None:
        dt_half = p.dt / 2.0
    if p.process_time_scheme == 'symmetric_fast_v3':
        return _symmetric_free_surface_step(eta, u, v, p, dt_half, F_rho_x, F_rho_y,
                                            column_face_transport)[0]
    if p.mode_split:
        ubt, vbt = u, v   # subcycle mode: caller passes BT velocity directly
    else:
        ubt, vbt = _barotropic_velocity(u, v, p)
        if p.column_geometry == 'nodal_dual_v1':
            column_divergence_offset = _column_divergence(u, v, p) - _reference_depth_divergence(ubt, vbt, p)

    F_x = jnp.zeros_like(ubt)
    F_y = jnp.zeros_like(vbt)
    if F_rho_x is not None:
        F_x = F_x + F_rho_x
        F_y = F_y + F_rho_y
    F_x = F_x + p.tau_x_2d / (RHO_0 * p.H_sw)
    F_y = F_y + p.tau_y_2d / (RHO_0 * p.H_sw)

    r_bt = p.r_bot if p.bottom_friction == 'linear' and p.column_geometry == 'legacy' else 0.0
    drag = 1.0 / (1.0 + r_bt * dt_half)
    # Mass-conserving (flux-form) divergence: face fluxes zeroed at wet/dry
    # interfaces so the divergence telescopes to zero over the wet domain. The
    # centered (roll) form leaks volume at coastlines and closed walls. Mask
    # ubt/vbt to wet first so the face averages carry no land values. (D2)
    if p.column_geometry == 'nodal_dual_v1':
        if column_face_transport is not None:
            column_transport_divergence = _face_transport_divergence(*column_face_transport, p)
        else:
            column_transport_divergence = _reference_depth_divergence(ubt, vbt, p)
            if column_divergence_offset is not None:
                column_transport_divergence = column_transport_divergence + column_divergence_offset
    else:
        div_bt = _divergence_conservative(ubt * p.wet_mask, vbt * p.wet_mask, p)
        column_transport_divergence = p.H_sw * div_bt

    # Forward-backward (Sielecki) free-surface coupling: update eta FIRST (old
    # velocity), then update barotropic momentum using the NEW eta gradient.
    # Forward-forward (both from the old state) is unconditionally unstable for
    # free gravity waves, |lambda| = sqrt(1+(dt*c*k)^2) > 1 for ALL k -- CFL does
    # not save forward-Euler. Forward-backward flips the trace to
    # 2 - dt^2*g*H*k^2 => |lambda| = 1 under CFL < 1, the standard OGCM
    # discretization (MOM6/ROMS/NEMO); bottom drag then decays the free mode.
    # (D22)
    if p.column_geometry == 'nodal_dual_v1':
        eta_new = eta - dt_half * column_transport_divergence
    else:
        eta_new = eta - dt_half * p.H_sw * div_bt

    eta_new = eta_new * p.wet_mask
    # Lateral sponge on eta (2D). No-op when sponge_rate_2d == 0.
    sw_decay = jnp.exp(-p.sponge_rate_2d * dt_half)
    # MASS-CONSERVING SPONGE: the bare decay eta *= sw_decay changes global
    # volume by dV = sum(A*eta*(sw_decay-1)) over the band, and the wind setup
    # makes the band-mean eta NEGATIVE on both hemispheres, so the sponge was
    # ADDING volume every step. Correct by returning the removed volume
    # UNIFORMLY over the wet domain: total volume is exactly conserved, the local
    # anomaly damping is unchanged, and a uniform eta offset has zero PGF so the
    # dynamics are untouched. (D23)

    # ── Semi-enclosed-sea eta relaxation (Mediterranean artifact fix) ──
    # Gibraltar (14 km wide) is sub-grid on a 1 deg mesh: the one-cell strait
    # cannot support the observed two-layer exchange, and the residual pressure
    # mismatch drives a spurious NET outflow that drains the basin linearly.
    # Standard OGCM remedy (MOM-family): Rayleigh-relax eta toward the basin
    # equilibrium INSIDE the sea only, eta *= exp(-eta_relax_rate*dt_half) where
    # eta_relax_mask = 1 (eta == 0 is the correct equilibrium: the basin-mean PGF
    # at Gibraltar then matches the Atlantic at the same latitude).
    # MASS-CONSERVING COMPENSATION: the same uniform-refill trick as the sponge
    # returns the removed volume over the global wet domain; a uniform eta offset
    # has zero PGF, so the relaxation damps the basin anomaly, not its water
    # mass. (D23)

    # Polar-cap filter: zonally average the poleward rows to kill the
    # cos(lat)->0 metric singularity (dx->0 makes the explicit SW CFL
    # unattainable at the edge).
    #
    # CRITICAL: average over WET points only and write back to WET points only.
    # The polar rows are ~30% land, and a naive all-column mean mixes ocean
    # (eta != 0) with land (eta == 0), forcing a zonally-uniform value that
    # creates a spurious PGF at EVERY coastline point in the band. (D21)
    def _cap(field2d):
        return _apply_polar_cap(field2d, p.wet_mask, p)

    # CONSISTENT-TRIPLE CAP: cap eta FIRST, then drive the barotropic momentum
    # update from the CAPPED eta's pressure gradient, then cap the resulting
    # ubt/vbt. Capping all three to independent zonal means leaves eta zonally
    # uniform but ubt/vbt carrying a different zonal structure, so the next
    # step's div(ubt) and grad(eta) are dynamically inconsistent. Capping eta
    # first makes ubt/vbt consistent with eta by construction; their cap is then
    # a CFL-safety smoothing, not an independent forcing. (D21)
    eta_new = _filter_barotropic_eta(eta_new, p, dt_half)
    # Energy-consistent PGF: the exact adjoint of the conservative divergence
    # (area-weighted), so the FB pair is neutral on the masked non-uniform grid.
    # The centered _d_dx/_d_dy gradient is NOT the adjoint and injects energy.
    if p.column_geometry == 'nodal_dual_v1':
        grad_eta_x, grad_eta_y = _reference_depth_gradient(eta_new, p)
    else:
        grad_eta_x, grad_eta_y = _gradient_conservative(eta_new, p)
    # Barotropic momentum with PGF + forcing (intermediate star state).
    u_star = ubt + dt_half * (-G_EARTH * grad_eta_x + F_x)
    v_star = vbt + dt_half * (-G_EARTH * grad_eta_y + F_y)
    # Semi-implicit barotropic Coriolis. The shallow-water momentum eqn is
    #   du/dt - f*v = -g*grad(eta) + F ;  dv/dt + f*u = -g*grad(eta) + F
    # Legacy backward Euler is stable but damps kinetic energy:
    #   u_new = (u_star + f*dt*v_star) / (1+(f*dt)^2)
    #   v_new = (v_star - f*dt*u_star) / (1+(f*dt)^2)
    # Without it the barotropic PGF has no geostrophic balance: it drives a
    # convergent ubt that grows eta monotonically. This is the barotropic
    # analogue of the 3D rotation in _linear_half_step. (D22)
    if p.process_time_scheme != 'legacy':
        half_rotation = p.f * (dt_half / 2.)
        right_x = u_star + half_rotation * vbt
        right_y = v_star - half_rotation * ubt
        denominator = 1. + half_rotation * half_rotation
        ubt_new = (right_x + half_rotation * right_y) / denominator
        vbt_new = (right_y - half_rotation * right_x) / denominator
    else:
        fd = p.f * dt_half
        denom = 1.0 + fd * fd
        ubt_new = (u_star + fd * v_star) / denom
        vbt_new = (v_star - fd * u_star) / denom
    ubt_new = ubt_new * drag
    vbt_new = vbt_new * drag

    ubt_new = ubt_new * p.wet_mask
    vbt_new = vbt_new * p.wet_mask
    ubt_new = ubt_new * sw_decay
    vbt_new = vbt_new * sw_decay
    ubt_new = _cap(ubt_new)
    vbt_new = _cap(vbt_new)

    if p.mode_split:
        # Subcycle mode: caller works on the 2D barotropic state directly
        # (eta, ubt, vbt); the single 3D<->BT projection happens ONCE per
        # baroclinic step in _step_impl, not per subcycle.
        return eta_new, ubt_new, vbt_new

    # Project barotropic delta back to 3D velocity (uniform over depth)
    delta_ubt = (ubt_new - ubt)[:, :, None]
    delta_vbt = (vbt_new - vbt)[:, :, None]
    projection_mask = p.wet_mask_z if p.column_geometry == 'nodal_dual_v1' else 1.
    u_new = u + delta_ubt * projection_mask
    v_new = v + delta_vbt * projection_mask
    # No-flux wall: enforce zero normal velocity at the N/S boundary rows on
    # the projected 3D v too, consistent with _step_impl's final mask.
    v_new = v_new * p.interior_mask_z
    return eta_new, u_new, v_new


def _rotate_baroclinic_shear(velocity_x, velocity_y, params, duration):
    """Rotate only the wet-depth anomaly; the fast mode rotates its own mean."""
    mean_x, mean_y = _barotropic_velocity(velocity_x, velocity_y, params)
    shear_x = (velocity_x - mean_x[..., None]) * params.wet_mask_z
    shear_y = (velocity_y - mean_y[..., None]) * params.wet_mask_z
    rotation = params.f
    if params.process_time_scheme == 'symmetric_fast_v3':
        rotation = rotation * params.interior_mask_z[..., 0]
    rotated_x, rotated_y = _coriolis_rotation_2d(shear_x, shear_y, rotation, duration)
    return (velocity_x + (rotated_x - shear_x) * params.wet_mask_z,
            velocity_y + (rotated_y - shear_y) * params.wet_mask_z)


def _linear_half_step(state, p, dt_half, budget=None, *, momentum_diffusion=None):
    """Linear half-step: FD diffusion + 2D Coriolis + free surface (or not, split).

    The default retains component-wise explicit diffusion; component CFL limits
    alone do not bound their combined update. An opt-in diffusion callback acts
    before masking, damping and rotation. The monolithic free surface remains
    forward Euler with CFL dt < dx/sqrt(gH). (D22)
    """
    # Explicit diffusion (horizontal Laplacian + vertical d2/dz2). nu_h CANNOT
    # be applied at dt_half in one explicit shot (nu_h*dt_half/dy^2 = 0.73 >
    # 0.25), but it MUST act on the full 3D baroclinic velocity: the N-step
    # residual subtracts nu_h*lap, so if L re-adds nothing the baroclinic SHEAR
    # (invisible to the depth-mean) loses all its damping. Apply nu_h here
    # subcycled (nu_sub_cyc substeps of dt_half/nu_sub_cyc, CFL 0.06 each) --
    # the same fixed-operator subcycling pattern as conv_nsub. Monolithic
    # dt=60-300 keeps the single-shot form (CFL-safe). (D12)
    if momentum_diffusion is not None:
        u, v = momentum_diffusion(state, p, dt_half)
    elif p.mode_split and p.n_subcyc > 0:
        # Subcycle count sized by the 0.5*LHS FTCS criterion. Legacy sizing
        # (nu_nsub=None) reuses n_subcyc; nu_nsub right-sizes it from the actual
        # metric worst case: nu_h*dt/(n*dy_min^2) <= 0.5 (see make_solver_global).
        # (D12)
        n_nu = max(1, int(p.n_subcyc if p.nu_nsub is None else p.nu_nsub))
        dts = dt_half / n_nu

        def _nu_sub(carry):
            cu, cv = carry
            return (cu + p.nu_h * _laplacian_h(cu, p) * dts,
                    cv + p.nu_h * _laplacian_h(cv, p) * dts), None

        (u, v), _ = _subcycle(_nu_sub, (state.u, state.v), n_nu, p)
    else:
        u = state.u + p.nu_h * _laplacian_h(state.u, p) * dt_half
        v = state.v + p.nu_h * _laplacian_h(state.v, p) * dt_half
    T = state.T + _horizontal_tracer_diffusion(state.T, p) * dt_half
    S = state.S + _horizontal_tracer_diffusion(state.S, p) * dt_half
    # Scale-selective biharmonic (nabla^4): damps grid-scale modes far more than
    # large-scale ones. Explicit forward-Euler; CFL nu_bi*dt/dx^4 < ~0.05.
    # docs/resolution_cfl_limits.md has the dx^4 auto-scaling.
    if p.nu_bi > 0.0 and momentum_diffusion is None:
        u = u - p.nu_bi * _biharmonic_h(state.u, p) * dt_half
        v = v - p.nu_bi * _biharmonic_h(state.v, p) * dt_half
    if p.kappa_bi > 0.0:
        T = T - p.kappa_bi * _horizontal_biharmonic_tracer(state.T, p) * dt_half
        S = S - p.kappa_bi * _horizontal_biharmonic_tracer(state.S, p) * dt_half
    if momentum_diffusion is None:
        u = u + _vertical_momentum_diffusion(state.u, p) * dt_half
        v = v + _vertical_momentum_diffusion(state.v, p) * dt_half
    T = T + _vertical_diffusion(state.T, _effective_kappa_v(p), p) * dt_half
    S = S + _vertical_diffusion(state.S, _effective_kappa_v(p), p) * dt_half
    # Mask: no diffusion updates over land or below seafloor (ghost water). Hold
    # land/ghost values at their ORIGINAL state (not zero): masking to zero
    # creates a T=0 cliff at every coastline that the (unmasked) _laplacian_h in
    # the N-step tracer residual reads as a huge gradient. Holding the pre-step
    # value preserves the no-flux (flat) land value the diffusive stencil already
    # assumes (mirror ghost = boundary value => no cliff). (D9)
    u = u * p.wet_mask_z + state.u * (1.0 - p.wet_mask_z)
    v = v * p.wet_mask_z + state.v * (1.0 - p.wet_mask_z)
    T = T * p.wet_mask_z + state.T * (1.0 - p.wet_mask_z)
    S = S * p.wet_mask_z + state.S * (1.0 - p.wet_mask_z)

    if budget is not None:
        diffusion_state = JaxStateG(u, v, T, S, state.eta, state.ice)
        budget.stage("linear_diffusion", state, diffusion_state)

    # Lateral sponge (Rayleigh damping): exponential decay, no damping CFL.
    # Applied in the linear half-step so Strang splitting gives total sponge time
    # = dt per full step. Absorbs the wind-driven barotropic energy that
    # Laplacian dissipation cannot arrest within its CFL cap, which otherwise
    # piles up at the polar edge rows.
    #
    # No-op when off: sponge_rate == 0 => decay == 1 and T_clim/S_clim == 0, so
    # every line below returns its input unchanged. When ON, T_clim/S_clim carry
    # the land sentinel on land and below-seafloor ghost cells, so the relaxation
    # leaves the land invariant the mask above restored intact. (D19, D23)
    decay = jnp.exp(-p.sponge_rate * dt_half)            # (nx,ny,1)
    u = u * decay
    v = v * decay
    T = p.T_clim_3d + (T - p.T_clim_3d) * decay
    S = p.S_clim_3d + (S - p.S_clim_3d) * decay

    if budget is not None:
        sponge_state = JaxStateG(u, v, T, S, state.eta, state.ice)
        budget.stage("sponge", diffusion_state, sponge_state)
        budget.sponge_sources(diffusion_state, decay)

    # Coriolis rotation (2D f-field, exact)
    if p.process_time_scheme != 'legacy':
        u, v = _rotate_baroclinic_shear(u, v, p, dt_half)
    else:
        u, v = _coriolis_rotation_2d(u, v, p.f, dt_half)

    # Semi-implicit free surface (with density barotropic PGF). mode_split: the
    # free surface runs ONCE per baroclinic step in _step_impl's barotropic
    # subcycle (dt_bt), not here -- the external-wave CFL forbids dt_half here.
    # The linear step keeps diffusion (nu_h subcycled above) + sponge + Coriolis
    # only. (D12, D22)
    if p.mode_split:
        return JaxStateG(u, v, T, S, state.eta, state.ice)
    F_rho_x, F_rho_y = _compute_bt_rho_pgf(state, p)
    eta, u, v = _free_surface_step_fd(state.eta, u, v, p, F_rho_x, F_rho_y, dt_half)
    updated = JaxStateG(u, v, T, S, eta, state.ice)
    if budget is not None:
        budget.stage("free_surface", sponge_state, updated)
    return updated


# ── Nonlinear explicit step (forward-backward RK2, FD) ─────────────

def _compute_tracer_residual(state, p, budget=None, face_transport=None, return_terms=False):
    """Tracer tendency minus the linear diffusion (handled by the linear step).

    The Strang split L(dt/2).N(dt).L(dt/2) handles ALL linear dissipation
    (Laplacian + biharmonic) in L, so the residual must SUBTRACT the linear terms
    the full tendency included or they are double-applied: dTdt -= kappa_h*lap and
    dTdt -= kappa_v*_d2_dz2 (the latter ran vertical diffusion at 2*kappa_v before
    the fix). The biharmonic is NOT in _compute_tracer_tendency (it is L-only), so
    it must not appear here at all -- a term here would be re-applied by N(dt) and
    cancel the L-step damping exactly (-dt/2 + dt - dt/2 = 0), a silent no-op. (D9)
    """
    result = _compute_tracer_tendency(state, p, budget=budget, face_transport=face_transport,
                                      return_terms=return_terms)
    dTdt, dSdt = result[:2]
    dTdt = dTdt - _horizontal_tracer_diffusion(state.T, p)
    dSdt = dSdt - _horizontal_tracer_diffusion(state.S, p)
    dTdt = dTdt - _vertical_diffusion(state.T, _effective_kappa_v(p), p)
    dSdt = dSdt - _vertical_diffusion(state.S, _effective_kappa_v(p), p)
    return (dTdt, dSdt, result[2]) if return_terms else (dTdt, dSdt)


def _compute_momentum_residual(state, p):
    """Momentum tendency minus linear parts (diffusion, Coriolis, bt PGF, bt wind).

    Biharmonic hyperviscosity is an L-only term (not in the full momentum tendency),
    so it must NOT appear here -- see _compute_tracer_residual for the no-op
    argument.
    """
    dudt, dvdt = _compute_momentum_tendency(state, p)
    # nu_h subtraction is unconditional: the full tendency ALWAYS includes
    # diff_h, so the residual must ALWAYS remove it -- the L step re-adds it.
    # Skipping the subtraction in split mode applied nu_h inside the N-step at
    # dt=3600, where nu_h*dt/dy^2 = 1.46 >> 0.25 exceeds the explicit CFL.
    # (D9, D12)
    dudt = dudt - p.nu_h * _laplacian_h(state.u, p)
    dvdt = dvdt - p.nu_h * _laplacian_h(state.v, p)
    # nu_v likewise: _linear_half_step applies nu_v*_d2_dz2 over dt/2 twice,
    # so omitting it here applied vertical momentum diffusion at 2*nu_v
    # (measured coefficient 2.12 per step).
    dudt = dudt - _vertical_momentum_diffusion(state.u, p)
    dvdt = dvdt - _vertical_momentum_diffusion(state.v, p)
    dudt = dudt - p.f[:, :, None] * state.v
    dvdt = dvdt + p.f[:, :, None] * state.u
    # barotropic PGF from eta. The full tendency's 3D PGF contains the eta part
    # via p_bt = RHO_0*G_EARTH*eta (top-node pressure, _compute_hydrostatic_pressure)
    # differentiated by _gradient_conservative_3d with PER-LAYER wet/ghost face
    # gating. mode_split: cancel it with the SAME operator on the SAME field, so
    # the subtraction is EXACT and the external mode acts on 3D momentum solely
    # through the subcycle projection (a bare or column-gated 2D gradient leaves a
    # spurious eta-PGF wherever the layer gate differs from the column gate).
    # Monolithic: keep the historical bare _d_dx/_d_dy subtraction bit-exact to
    # the pre-split solver. (D12)
    if p.column_geometry == 'nodal_dual_v1':
        eta_pgf_x, eta_pgf_y = _reference_depth_gradient(state.eta, p)
        dudt = dudt + G_EARTH * eta_pgf_x[..., None]
        dvdt = dvdt + G_EARTH * eta_pgf_y[..., None]
    elif p.mode_split:
        eta_pgf_x, eta_pgf_y = _gradient_conservative_3d(state.eta[:, :, None], p)
        dudt = dudt + G_EARTH * eta_pgf_x
        dvdt = dvdt + G_EARTH * eta_pgf_y
    else:
        dudt = dudt + G_EARTH * _d_dx(state.eta[:, :, None], p)
        dvdt = dvdt + G_EARTH * _d_dy(state.eta[:, :, None], p)
    # barotropic PGF from density anomaly
    bt_rho_x, bt_rho_y = _compute_bt_rho_pgf(state, p)
    dudt = dudt - bt_rho_x[:, :, None]
    dvdt = dvdt - bt_rho_y[:, :, None]
    # barotropic wind
    bt_wind_x = p.tau_x_2d / (RHO_0 * p.H_sw)
    bt_wind_y = p.tau_y_2d / (RHO_0 * p.H_sw)
    dudt = dudt - bt_wind_x[:, :, None]
    dvdt = dvdt - bt_wind_y[:, :, None]
    # Mask the residual by wet_mask_z for consistency with the full tendency.
    # _compute_momentum_tendency masks dudt to 0 in ghost water, but the
    # barotropic subtractions above broadcast uniformly over ALL depths, so in
    # ghost layers the residual was -bt_pgf (a spurious PGF) instead of 0.
    # Masking closes the mismatch so the N-step residual equals the masked full
    # tendency minus its linear parts everywhere. (D12)
    dudt = dudt * p.wet_mask_z
    dvdt = dvdt * p.wet_mask_z
    return dudt, dvdt


def _tracer_rk_subcycle(state, params, face_transport=None, budget=None):
    """Heun substeps of the instantaneous nonlinear RHS, with actual RK weights."""
    subcycles = max(int(params.adv_nsub), int(params.conv_nsub))
    subparams = params._replace(dt=params.dt / subcycles, adv_nsub=1, conv_nsub=1)

    def advance(current):
        temperature, salinity = current
        stage = state._replace(T=temperature, S=salinity)
        first = _compute_tracer_residual(stage, subparams, face_transport=face_transport,
                                         return_terms=budget is not None)
        predicted = stage._replace(T=temperature + subparams.dt * first[0],
                                    S=salinity + subparams.dt * first[1])
        second = _compute_tracer_residual(predicted, subparams, face_transport=face_transport,
                                          return_terms=budget is not None)
        updated = (temperature + 0.5 * subparams.dt * (first[0] + second[0]),
                   salinity + 0.5 * subparams.dt * (first[1] + second[1]))
        terms = (jax.tree.map(lambda start, end: 0.5 * (start + end), first[2], second[2])
                 if budget is not None else None)
        return updated, terms

    (temperature, salinity), terms = _subcycle(advance, (state.T, state.S), subcycles, params)
    if budget is not None:
        budget.tracer_transport(face_transport if face_transport is not None
                                else _layer_face_transports(state.u, state.v, params), weight=1.)
        budget.surface_sources(terms[0], duration=params.dt)
        budget.nonlinear_terms(terms[1], duration=params.dt, absolute_tendencies=terms[4])
        budget.advection_boundary_fluxes(terms[2], terms[3], duration=params.dt)
    return state._replace(T=temperature, S=salinity)


def _nonlinear_predictor_rk2(state, params, duration, budget=None):
    """Two actual momentum stages; accepted tracer transport is replayed later."""
    tracer = _tracer_rk_subcycle(state, params._replace(dt=duration), budget=budget)
    first_x, first_y = _compute_momentum_residual(state, params)
    implicit_drag = params.mode_split and params.bottom_friction == 'linear'
    bottom = params.bottom_mask * params.wet_mask_z
    if implicit_drag:
        first_x = first_x + params.r_bot * state.u * bottom
        first_y = first_y + params.r_bot * state.v * bottom
    predicted = tracer._replace(u=state.u + duration * first_x, v=state.v + duration * first_y)
    second_x, second_y = _compute_momentum_residual(predicted, params)
    if implicit_drag:
        second_x = second_x + params.r_bot * predicted.u * bottom
        second_y = second_y + params.r_bot * predicted.v * bottom
    velocity_x = state.u + 0.5 * duration * (first_x + second_x)
    velocity_y = state.v + 0.5 * duration * (first_y + second_y)
    if implicit_drag and params.process_time_scheme != 'symmetric_fast_v3':
        decay = jnp.exp(-params.r_bot * duration * bottom)
        velocity_x, velocity_y = velocity_x * decay, velocity_y * decay
    return tracer._replace(u=velocity_x, v=velocity_y)


def _explicit_full_step(state, p, dt, budget=None):
    """Forward-backward RK2 for nonlinear tendencies (FD).

    Tracers updated first (old velocity), then momentum uses predicted T
    for the baroclinic PGF — shifts internal-wave eigenvalues left of the
    imaginary axis for neutral stability.
    """
    if p.process_time_scheme in ('subcycled_rk2_v2', 'symmetric_fast_v3'):
        return _nonlinear_predictor_rk2(state, p, dt, budget=budget)
    dT1, dS1 = _compute_tracer_residual(state, p, budget=budget)
    T_pred = state.T + dT1 * dt
    S_pred = state.S + dS1 * dt
    state_T = JaxStateG(state.u, state.v, T_pred, S_pred, state.eta, state.ice)

    du1, dv1 = _compute_momentum_residual(state_T, p)
    # ── Semi-implicit linear bottom drag (split mode only) ──
    # The residual's -r_bot*u is forward-Euler here: pure-drag amplitude
    # 1 - r*dt flips past |1-r*dt| <= 1 at r*dt > 2, and the mode-split dt=3600
    # gives r*dt = 3.6, so the bottom layer amplifies with sign alternation. Both
    # RK2 stages evaluate the residual at the OLD velocity, so the drag part is
    # identical in du1/du2: strip it from both and apply the EXACT linear-drag
    # decay exp(-r_bot*dt) after the RK2 update -- unconditionally stable and
    # exact for the pure-drag sub-operator. The monolithic path is untouched (its
    # dt range keeps the explicit form stable). (D22)
    implicit_drag = p.mode_split and p.bottom_friction == 'linear'
    if implicit_drag:
        bm = p.bottom_mask * p.wet_mask_z
        du1 = du1 + p.r_bot * state.u * bm
        dv1 = dv1 + p.r_bot * state.v * bm
    u_pred = state.u + du1 * dt
    v_pred = state.v + dv1 * dt
    # ── Tracer stage-2 advection velocity ──
    # RK2 evaluates the momentum residual at the OLD velocity in BOTH stages; the
    # tracer stage-2 advection was the lone exception, using the raw forward-Euler
    # predictor u_pred, a velocity the momentum scheme itself never reaches. That
    # velocity is strongly divergent (baroclinic PGF undamped over dt at the
    # predictor) and the flux-form advection of it leaks column heat at rate
    # -sum(AREA*Fz[0]*T[0]). Freezing it makes the tracer RK2 consistent with the
    # momentum RK2 (both stages at the old velocity) and changes the final T field
    # by only 0.037 K rms. Default False = historical behavior. (D13)
    if p.freeze_adv_vel:
        state_pred = JaxStateG(state.u, state.v, T_pred, S_pred, state.eta, state.ice)
    else:
        # Column-divergence-consistent stage-2 velocity (project_adv_vel): the
        # raw predictor's column divergence is O(dt) and drives the flux-form
        # column heat leak; project it onto the divergence-free column space (the
        # same space the barotropic subcycle later enforces on the FINAL u) so
        # Fz_top = Fz[0]*T[0] conserves column heat. No-op when off. (D13)
        if p.project_adv_vel:
            u_adv, v_adv = _project_column_divergence(u_pred, v_pred, p, dt)
            if budget is not None:
                budget.column_projection(u_pred, v_pred, u_adv, v_adv)
        else:
            u_adv, v_adv = u_pred, v_pred
        state_pred = JaxStateG(u_adv, v_adv, T_pred, S_pred, state.eta, state.ice)

    dT2, dS2 = _compute_tracer_residual(state_pred, p, budget=budget)
    T_new = state.T + 0.5 * (dT1 + dT2) * dt
    S_new = state.S + 0.5 * (dS1 + dS2) * dt
    state_T_new = JaxStateG(state.u, state.v, T_new, S_new, state.eta, state.ice)

    du2, dv2 = _compute_momentum_residual(state_T_new, p)
    if implicit_drag:
        du2 = du2 + p.r_bot * state.u * bm
        dv2 = dv2 + p.r_bot * state.v * bm
    u_new = state.u + 0.5 * (du1 + du2) * dt
    v_new = state.v + 0.5 * (dv1 + dv2) * dt
    if implicit_drag:
        decay = jnp.exp(-p.r_bot * dt * bm)     # 1 away from the bottom
        u_new = u_new * decay
        v_new = v_new * decay
    return JaxStateG(u_new, v_new, T_new, S_new, state.eta, state.ice)


def _polar_cap_weights(ncap, ntaper):
    """Blend weights for the tapered polar cap, in POLE-INWARD order.

    1D array of length ncap+ntaper, the blend fraction of the zonal mean applied to
    each row from the pole inward: rows [0:ncap] get weight 1.0 (full zonal average,
    killing the cos(lat)->0 metric singularity), rows [ncap:ncap+ntaper] ramp 1->0
    via a cos^2 taper. A hard cutoff (ntaper=0) leaves a cliff at j=ncap that _d_dy
    amplifies exponentially. (D21)

    The order is a caller contract, not a convenience: the weight at index 0
    belongs on the POLE ROW. The south band slices pole-inward and uses this
    array as-is; the north band slices pole-LAST and must reverse it. Applying
    the north band unflipped leaves the wall row effectively uncapped -- see
    _apply_polar_cap.
    """
    if ncap <= 0:
        return jnp.zeros(0)
    full = jnp.ones(ncap)
    if ntaper <= 0:
        return full
    # cos^2 ramp from 1 (at the cap edge) to 0 (into the interior).
    x = jnp.linspace(0.0, jnp.pi / 2.0, ntaper + 2)[1:-1]   # open interval
    ramp = jnp.cos(x) ** 2
    return jnp.concatenate([full, ramp])


def _apply_polar_cap(field, wm, p):
    """Tapered wet-point zonal-average polar cap on a 2D or 3D field.

    Blends the wet-point zonal mean of the poleward rows in over polar_cap_taper
    transition rows with a cos^2 ramp (a hard cutoff is a meridional cliff _d_dy
    amplifies). Land and ghost nodes keep their masked value, so wm must match the
    field's rank: the 3D mask matters, since the 2D one is column-wide and would let
    ghost nodes below a shallow seafloor into the deep-level mean. (D21)

    Non-legacy float32 candidates reduce and blend only the cap bands in float64,
    then return the original dtype. Incremental blending preserves constant wet
    fields exactly. This does not make rounded tracer inventories conservative.
    """
    ncap = p.polar_cap_rows
    if ncap <= 0:
        return field
    nb = ncap + p.polar_cap_taper
    # Runtime-built jnp.ones/linspace are float64 under jax_enable_x64; cast
    # to the field's dtype or the blend promotes the whole field to f64
    # (silently defeats the fp32 state/params cast downstream).
    wts = _polar_cap_weights(ncap, p.polar_cap_taper).astype(field.dtype)

    def _cap_band(f, w, wts_band):
        if field.dtype == jnp.float32 and getattr(p, 'process_time_scheme', 'legacy') != 'legacy':
            wet64 = w.astype(jnp.float64)
            values64 = f.astype(jnp.float64) * wet64
            count64 = jnp.maximum(jnp.sum(wet64, axis=0, keepdims=True), 1.)
            mean64 = jnp.sum(values64, axis=0, keepdims=True) / count64
            fraction64 = wts_band.astype(jnp.float64).reshape((1, nb) + (1,) * (field.ndim - 2))
            return (values64 + fraction64 * (mean64 * wet64 - values64)).astype(field.dtype)
        # Wet-point zonal mean over the FULL band (one value per (row, ...)),
        # broadcast back; land/ghost stays at its masked value.
        s = f * w
        wsum = jnp.maximum(jnp.sum(w, axis=0, keepdims=True), 1.0)
        zmean = jnp.sum(s, axis=0, keepdims=True) / wsum
        zmean = jnp.broadcast_to(zmean, f.shape) * w
        wb = wts_band.reshape((1, nb) + (1,) * (field.ndim - 2))
        return wb * zmean + (1.0 - wb) * s

    # _polar_cap_weights orders the blend pole-inward (weight 1 on the POLE
    # row, tapering into the interior). The south band already slices inward
    # from its pole, so the order matches; the north band slices POLE-FIRST
    # (j=ny-1 back toward the interior), so its weights must be flipped.
    # Unflipped, the north cap spends its full zonal mean nb-1 rows INSIDE
    # the wall and leaves the wall row itself at weight ~0.15 of it -- i.e.
    # effectively uncapped. The wall row then grows a 2dx zonal checkerboard
    # (measured on the real ETOPO relief: peak |u| pinned to j=ny-1, 1.0 ->
    # 24.8 m/s between day 1.0 and day 2.0). (D21)
    south = _cap_band(field[:, :nb], wm[:, :nb], wts)
    north = _cap_band(field[:, -nb:], wm[:, -nb:], jnp.flip(wts))
    return jnp.concatenate([south, field[:, nb:-nb], north], axis=1)


def nu_nsub_for_2d_cfl(nu_h, dt, dx_2d, dy, margin=0.25):
    """Subcycle count that keeps the explicit nu_h Laplacian inside its FTCS bound.

    The 5-point Laplacian's most negative eigenvalue sums BOTH metric terms, so
    the split L half-step is stable iff
    ``nu_h * dt_sub * (1/dx^2 + 1/dy^2) <= margin`` with ``dt_sub = dt/(2*n)``.
    The worst point is the ZONAL spacing at the highest latitude -- ``dx =
    dy*cos(lat)``, so at lat_max=60 the zonal spacing is HALF the meridional one
    and its term is 3.9x the meridional one in the sum. Sizing from ``dy`` alone
    (a scalar) misses that: at nu_h=5e6, dt=3600, lat_max=60 it returned 6. The
    bound above is deliberately conservative -- the REALIZED most negative
    eigenvalue of this stencil on the ETOPO metric is 1.4515e-9, 91.9% of the
    sum-of-worst-cases 1.5794e-9 -- so the growth threshold is nu_h*dts*|lam|
    > 2, i.e. LHS > 0.544, i.e. n >= 6.53 (7). The old formula's 6 sits 9% over
    that line: |1 + nu_h*dts*lam| = 1.177 per substep, 2.66x per half-step.
    This returns 15 (factor 0.13, i.e. strongly damped).

    The polar cap hides the symptom by zonally averaging exactly those rows, so
    the undersizing stays latent while the cap does its job: the cap takes the
    n=6 spectral radius from 2.66 to 1.00 (the strictly neutral modes), and
    mis-anchoring its north band leaves 1.53 -- the growth that was measured as
    FAIL_BLOWUP at step 36 with the peak pinned to j=ny-1. At the production
    count (n=24) the same radius is 1.00 with NO cap at all, which is why the
    100-yr runs were stable for their own reason and not by accident. (D21)

    Returns the count for the split path; the monolithic path applies nu_h once
    per half-step and is sized by dt=60-300 instead. (D12)
    """
    inv_dx2_max = float(np.max(np.asarray(dx_2d) ** -2))   # smallest dx wins
    inv_dy2 = 1.0 / float(dy) ** 2
    return max(1, int(np.ceil(nu_h * (dt / 2.0)
                               * (inv_dx2_max + inv_dy2) / margin)))


def _dynamic_ice_closure(state, p, budget=None):
    """Advance the minimal stateful ice closure after one dynamics step.

    This opt-in first-order surface operator applies atmospheric heat once.
    Water sensible heat minus ice latent heat is conserved through phase change.
    Existing ice exchanges heat at the surface node; excess melt energy and
    open-water flux are distributed over the wet mixed-layer overlap. There is
    no ice dynamics, entrainment or resolved ice thermodynamic column.
    """
    if not getattr(p, 'dynamic_ice', False):
        return state

    rho_ice = 917.0
    latent_heat = 3.34e5
    ice_salt_diff = 30.0
    dt = float(p.dt)

    T_sst = state.T[:, :, 0]
    ice = jnp.maximum(jnp.broadcast_to(
        jnp.asarray(state.ice, dtype=state.T.dtype), (p.nx, p.ny)), 0.0)

    insulation = 1.0 / (1.0 + ice / p.ice_insulation_scale_m)
    air_minus_sst = p.T_atm_3d[:, :, 0] - T_sst
    q = (p.Q_heat_2d
         + p.lambda_bulk * air_minus_sst
         + p.coastal_bulk_lambda_2d * air_minus_sst)
    q = q * insulation * p.wet_mask

    weights = _surface_heat_weights(p)
    heat_capacity = RHO_0 * C_P * p.dz_surface
    energy = q * dt
    projected = state.T + energy[:, :, None] * weights / (RHO_0 * C_P)
    freeze_deficit = heat_capacity * jnp.maximum(
        p.ice_freeze_temp_c - projected[:, :, 0], 0.)
    open_ice = freeze_deficit / (rho_ice * latent_heat)
    open_temperature = projected.at[:, :, 0].set(
        jnp.maximum(projected[:, :, 0], p.ice_freeze_temp_c))

    enthalpy = (heat_capacity * (T_sst - p.ice_freeze_temp_c)
                - rho_ice * latent_heat * ice + energy)
    existing_ice = jnp.maximum(-enthalpy, 0.) / (rho_ice * latent_heat)
    existing_temperature = state.T.at[:, :, 0].set(p.ice_freeze_temp_c)
    existing_temperature = (existing_temperature
                             + jnp.maximum(enthalpy, 0.)[:, :, None]
                             * weights / (RHO_0 * C_P))
    ice_new = jnp.where(ice > 0., existing_ice, open_ice) * p.wet_mask
    temperature = jnp.where((ice > 0.)[:, :, None], existing_temperature, open_temperature)
    salt_change = (rho_ice * ice_salt_diff / RHO_0
                   * (ice_new - ice)[:, :, None] * weights)
    T = jnp.where(p.wet_mask_z > 0.5, temperature, state.T)
    S = jnp.where(p.wet_mask_z > 0.5, state.S + salt_change, state.S)
    if budget is not None:
        budget.ice_sources(q, ice_new - ice, rho_ice, ice_salt_diff)
    return JaxStateG(state.u, state.v, T, S, state.eta,
                     ice_new * p.wet_mask)


def _barotropic_subcycle_transport(state, params):
    """Return actual OLD-face time mean and eta changes not caused by transport.

    The shear is frozen at the post-L/N/L predictor. Each OLD barotropic value
    is lifted onto that shear before recording the open-face volume flux. The
    optional transport-matched path drives eta with those same faces. Filtering,
    sponge and eta relaxation are recorded separately, not fitted into a flux.
    """
    if params.process_time_scheme == 'symmetric_fast_v3':
        state = state._replace(v=state.v * params.interior_mask_z)
    forcing_x, forcing_y = _compute_bt_rho_pgf(state, params)
    initial_u, initial_v = _barotropic_velocity(state.u, state.v, params)
    offset = (_column_divergence(state.u, state.v, params)
              - _reference_depth_divergence(initial_u, initial_v, params))
    zero = jnp.zeros_like(state.eta)

    def advance(carry):
        eta, mean_u, mean_v, total_x, total_y, filter_change = carry
        velocity_x = state.u + (mean_u - initial_u)[..., None] * params.wet_mask_z
        velocity_y = state.v + (mean_v - initial_v)[..., None] * params.wet_mask_z
        layers = _layer_face_transports(velocity_x, velocity_y, params)
        faces = tuple(jnp.sum(flux, axis=-1) for flux in layers)
        if params.match_barotropic_transport:
            divergence = _face_transport_divergence(*faces, params)
        else:
            divergence = _reference_depth_divergence(mean_u, mean_v, params) + offset
        if params.process_time_scheme == 'symmetric_fast_v3':
            updated, faces = _symmetric_free_surface_step(
                eta, mean_u, mean_v, params, params.dt_bt, forcing_x, forcing_y, faces)
            divergence = _face_transport_divergence(*faces, params)
        else:
            updated = _free_surface_step_fd(
                eta, mean_u, mean_v, params, forcing_x, forcing_y, dt_half=params.dt_bt,
                column_divergence_offset=offset,
                column_face_transport=faces if params.match_barotropic_transport else None)
        eta_new, mean_u_new, mean_v_new = updated
        nontransport_change = eta_new - (eta - params.dt_bt * divergence)
        return (eta_new, mean_u_new, mean_v_new, total_x + faces[0], total_y + faces[1],
                filter_change + nontransport_change), None

    final, _ = _subcycle(advance, (state.eta, initial_u, initial_v, zero, zero, zero), int(params.n_subcyc), params)
    eta, mean_u, mean_v, total_x, total_y, filter_change = final
    column_transport = (total_x / params.n_subcyc, total_y / params.n_subcyc)
    updated = state._replace(eta=eta,
                             u=state.u + (mean_u - initial_u)[..., None] * params.wet_mask_z,
                             v=state.v + (mean_v - initial_v)[..., None] * params.wet_mask_z)
    return updated, column_transport, filter_change


def _tracer_step_with_transport(state, params, column_transport, budget=None, advection_velocity=None):
    """Replay only the accepted tracer stages with the actual fast-mode mean.

    Both RK stages and every advection substep share matched open-layer faces.
    The provisional tracer update used for momentum forcing is discarded; its
    sources are not booked. This is a lagged predictor/corrector, not a claim of
    second-order coupled momentum or a moving-volume inventory formulation.
    """
    velocity_x, velocity_y = (state.u, state.v) if advection_velocity is None else advection_velocity
    faces = _match_layer_face_transports(velocity_x, velocity_y, column_transport, params)
    if params.process_time_scheme in ('subcycled_rk2_v2', 'symmetric_fast_v3'):
        return _tracer_rk_subcycle(state, params, face_transport=faces, budget=budget)
    temperature_1, salinity_1 = _compute_tracer_residual(state, params, budget=budget, face_transport=faces)
    predicted = state._replace(T=state.T + params.dt * temperature_1,
                               S=state.S + params.dt * salinity_1)
    temperature_2, salinity_2 = _compute_tracer_residual(predicted, params, budget=budget, face_transport=faces)
    return state._replace(T=state.T + 0.5 * params.dt * (temperature_1 + temperature_2),
                          S=state.S + 0.5 * params.dt * (salinity_1 + salinity_2))


def _linear_bottom_drag_step(state, params, duration, budget=None):
    """Exact wet-bottom drag, separately audited from numerical face filtering."""
    if params.bottom_friction != 'linear' or params.r_bot == 0.:
        return state
    decay = jnp.exp(-params.r_bot * duration * params.bottom_mask * params.wet_mask_z)
    updated = state._replace(u=state.u * decay, v=state.v * decay)
    if budget is not None:
        budget.bottom_drag(state, updated)
    return updated


def _step_impl(state, p, budget=None):
    """Strang splitting: L(dt/2) -> N(dt) -> L(dt/2).

    Default mode_split=True retains the L/N/L baroclinic core; linear half-steps
    carry diffusion/sponge/Coriolis, including subcycled nu_h, but no free surface. After the
    second L half-step, n_subcyc barotropic forward-backward subcycles of dt_bt
    evolve (eta, ubt, vbt), driven by the rho-PGF + wind forcing computed from the
    post-L/N/L predictor, held fixed over the subcycle. The final (ubt, vbt) is projected back onto the 3D
    velocity as a uniform-in-depth delta. The opt-in transport candidate replays
    accepted tracer stages with the actual fast-mode mean; momentum retains its
    provisional tracer forcing. Static nodal inventories remain approximate. (D12)
    """
    dt_half = p.dt / 2.0
    # Capture land/ghost values BEFORE the step. Final masking holds these
    # (not zero): masking land T to 0 builds a coastline cliff that the
    # unmasked _laplacian_h in the next step's tracer residual reads as a
    # huge gradient, seeding an exponentially-growing spurious PGF.
    land_u, land_v = state.u, state.v
    land_T, land_S = state.T, state.S
    if p.process_time_scheme == 'symmetric_fast_v3':
        state = _linear_bottom_drag_step(state, p, dt_half, budget=budget)
    state = _linear_half_step(state, p, dt_half, budget=budget)
    nonlinear_start = state
    predictor_budget = None if p.match_barotropic_transport else budget
    state = _explicit_full_step(state, p, p.dt, budget=predictor_budget)
    nonlinear_end = state
    if predictor_budget is not None:
        budget.stage("nonlinear", nonlinear_start, state)
    state = _linear_half_step(state, p, dt_half, budget=predictor_budget)
    barotropic_start = state

    # ── mode split: barotropic subcycle (free surface + nu_h) ──
    # Runs AFTER the L/N/L core on the un-masked final 3D state (before the
    # polar cap / land hold below, so the subcycle sees the same masked,
    # capped input the monolithic path fed _free_surface_step_fd).
    if p.mode_split and p.column_geometry == 'nodal_dual_v1' and (budget is not None or p.match_barotropic_transport):
        state, column_transport, filter_change = _barotropic_subcycle_transport(state, p)
        if budget is not None:
            budget.stage("free_surface", barotropic_start, state)
            budget.barotropic_transport(barotropic_start.eta, state.eta, column_transport, filter_change)
        if p.match_barotropic_transport:
            advection_velocity = ((0.5 * (nonlinear_start.u + nonlinear_end.u),
                                   0.5 * (nonlinear_start.v + nonlinear_end.v))
                                  if p.process_time_scheme in ('subcycled_rk2_v2', 'symmetric_fast_v3') else None)
            tracer_state = _tracer_step_with_transport(nonlinear_start, p, column_transport, budget=budget,
                                                      advection_velocity=advection_velocity)
            if budget is not None:
                budget.stage("nonlinear", nonlinear_start, tracer_state)
            tracer_state = _linear_half_step(tracer_state, p, dt_half, budget=budget)
            state = state._replace(T=tracer_state.T, S=tracer_state.S)
        if budget is not None:
            budget.before_transport_filter(state)
    elif p.mode_split:
        # Coupling forcing from the baroclinic state, held fixed over the
        # subcycle (MOM-style forcing lag at dt=3600 s is standard).
        F_rho_x, F_rho_y = _compute_bt_rho_pgf(state, p)
        ubt0, vbt0 = _barotropic_velocity(state.u, state.v, p)
        column_divergence_offset = None
        if p.column_geometry == 'nodal_dual_v1':
            column_divergence_offset = (_column_divergence(state.u, state.v, p)
                                        - _reference_depth_divergence(ubt0, vbt0, p))
        ubt, vbt = ubt0, vbt0
        eta = state.eta

        # lax.scan barotropic subcycle (when use_scan): one fused device-side loop
        # of n_subcyc forward-backward pairs instead of n_subcyc unrolled copies.
        # Numerically IDENTICAL; scan just compiles the body once, cutting XLA
        # graph size and host launch overhead. The Python-loop path stays default
        # (None) = bit-exact legacy trace.
        def _fb(carry):
            # One FB pair per call = dt_bt of evolution (caller passes
            # dt_half=dt_bt); each subcycle feeds the 2D triple forward.
            eta_i, ubt_i, vbt_i = carry
            return _free_surface_step_fd(
                eta_i, ubt_i, vbt_i, p, F_rho_x, F_rho_y, dt_half=p.dt_bt,
                column_divergence_offset=column_divergence_offset), None

        (eta, ubt, vbt), _ = _subcycle(_fb, (eta, ubt, vbt), int(p.n_subcyc), p)
        # Project the subcycle's net barotropic delta onto the 3D velocity
        # (uniform over depth) — same projection as the monolithic path.
        delta_ubt = (ubt - ubt0)[:, :, None]
        delta_vbt = (vbt - vbt0)[:, :, None]
        projection_mask = p.wet_mask_z if p.column_geometry == 'nodal_dual_v1' else 1.
        state = JaxStateG(state.u + delta_ubt * projection_mask, state.v + delta_vbt * projection_mask,
                          state.T, state.S, eta, state.ice)
        if budget is not None:
            budget.stage("free_surface", barotropic_start, state)

    if p.process_time_scheme == 'symmetric_fast_v3':
        state = _linear_bottom_drag_step(state, p, dt_half, budget=budget)
        if budget is not None:
            budget.before_transport_filter(state)

    # 3D polar-cap filter: zonally average the cap rows of u,v,T,S to kill
    # the cos(lat)->0 metric singularity in the diffusion/advection operators.
    u = _apply_polar_cap(state.u, p.wet_mask_z, p)
    v = _apply_polar_cap(state.v, p.wet_mask_z, p)
    T = _apply_polar_cap(state.T, p.wet_mask_z, p)
    S = _apply_polar_cap(state.S, p.wet_mask_z, p)
    # Final mask enforcement: hold land/ghost at the pre-step value (no cliff).
    wmask = p.wet_mask_z
    u = u * wmask + land_u * (1.0 - wmask)
    v = v * wmask + land_v * (1.0 - wmask)
    T = T * wmask + land_T * (1.0 - wmask)
    S = S * wmask + land_S * (1.0 - wmask)
    # No-flux wall: zero the NORMAL (meridional) velocity at the N/S boundary
    # rows so no flow crosses the closed wall. Combined with the mirror-ghost
    # stencils in _d_dy/_laplacian_h (zero normal gradient), this is the full
    # closed-boundary condition that replaces the unstable one-sided stencil.
    v = v * p.interior_mask_z
    masked = JaxStateG(u, v, T, S, state.eta, state.ice)
    if budget is not None:
        budget.stage("polar_cap_and_masks", state, masked)
    updated = _dynamic_ice_closure(masked, p, budget=budget)
    if budget is not None:
        budget.stage("dynamic_ice", masked, updated)
        budget.after_transport_filter(updated)
    return updated


# ── Public API ──────────────────────────────────────────────────────

def make_solver_global(grid, physics, dt, forcing=None,
                       T_atm=None, lambda_bulk=0.0,
                       mixed_layer_depth_m=None, mixed_layer_mask=None,
                       mixed_layer_depth_2d=None,
                       ice_freeze_temp_c=-1.8, ice_salt_flux=0.0,
                       dynamic_ice=False, ice_insulation_scale_m=1.0,
                       S_ref_surf=None, sss_restore_days=0.0,
                       coastal_restore_mask=None, coastal_restore_days=0.0,
                       coastal_bulk_mask=None, coastal_bulk_lambda=0.0,
                       coastal_kappa_h_mask=None, coastal_kappa_h=0.0,
                       coastal_kappa_v_mask=None, coastal_kappa_v=0.0,
                       coastal_restore_T=None,
                       sponge_days=0.0, sponge_cells=0,
                       T_init=None, S_init=None,
                       polar_cap_rows=2, polar_cap_taper=3, return_params=False,
                       eta_relax_days=0.0, eta_relax_box=None,
                       eta_relax_buffer=1.0, dynamic_forcing=False,
                       mode_split=False, dt_bt=150.0, nu_nsub=None,
                       dtype='float64', use_scan=False, freeze_adv_vel=False,
                       conservative_kv=False, project_adv_vel=False,
                       localize_conv=False, monotone_adv=False,
                       fct_adv=False, projection_niter=None,
                       projection_rtol=None, projection_preconditioner='none',
                       projection_max_refinements=2, column_geometry='legacy',
                       match_barotropic_transport=False, process_time_scheme='legacy'):
    """Create a JIT-compiled global FD ocean solver.

    Key properties:
      - Pure finite-difference operators; no wavenumber / decay factors. Forcing is
        physical-space 2D (tau_x, tau_y, Q_heat); no pre-FFT.
      - Free surface is forward-backward (Sielecki) + polar-cap filter, so dt must
        sit below the external-gravity-wave CFL unless mode_split=True (barotropic
        subcycling at dt_bt; off = bit-exact to the pre-split solver). (D12, D22)
      - Lateral sponge at the POLAR EDGE rows: the y axis is a closed boundary, not
        a periodic seam, so wind-driven barotropic energy piles up there. (D19, D23)
      - No SST restore (bulk flux only). Surface SALINITY restoring is available
        (S_ref_surf + sss_restore_days>0): Haney relaxation of SSS to climatology,
        mirroring the bulk-heat-flux form. Optional eta_relax adds a mass-conserving
        Rayleigh SSH relaxation inside a semi-enclosed sea. (D23, D24)
      - nu_nsub: None = legacy nu_h subcycle count (n_subcyc); 'cfl' = right-sized
        from the explicit-diffusion CFL; int = verbatim. (D12)
      - dtype='float32' casts params/state to fp32 (1:64 FP64:FP32 on Ada GPUs --
        the biggest kernel-time lever); default 'float64' = bit-exact.
      - use_scan=True runs the barotropic subcycle as lax.scan (numerically
        identical, smaller XLA graph / fewer host launches).
      - projection_niter overrides OCEAN_PAV_NITER, resolved once at construction.
        projection_rtol has a dtype floor; optional Jacobi scales the exact native
        Poisson diagonal. projection_max_refinements bounds actual-transport
        corrections with an original-RHS stopping floor. Neither a cap nor
        refinement is a convergence or physical conservation guarantee.
      - monotone_adv=True switches horizontal tracer face values to first-order
        donor-cell (upwind). ``fct_adv=True`` instead uses a local bounded
        centered face value; it takes precedence over monotone_adv.
      - column_geometry='nodal_dual_v1' is an opt-in M1 repair candidate:
        static nodal dual cells, an explicit staircase/deep truncation policy,
        common-wet depth fluxes and their volume-weighted adjoint. Requires
        conservative_kv/localize_conv. It does not yet match the actual fast-mode
        time-averaged tracer transport or define true moving-volume inventories.
        Default 'legacy' retains the frozen original numerical path.
      - match_barotropic_transport=True is an opt-in M2 transport candidate,
        requiring mode_split and nodal_dual_v1. Retains the original momentum
        predictor; replays accepted tracer stages with the actual OLD-face
        barotropic time mean. No source is counted twice. Retains static nodal
        thickness and the original surface tracer flux; no exact moving-volume
        heat/salt conservation, full temporal order or speedup is claimed.
      - process_time_scheme='consistent_split_v1' is an opt-in M3 candidate:
        convection subcycles retain physical kappa; slow rotation acts only on
        shear, and fast mean rotation uses implicit midpoint. Requires M2.
        The forward-backward gravity update and other process time errors remain;
        this is not a claim of second-order accuracy for the complete step.
      - process_time_scheme='subcycled_rk2_v2' additionally uses actual Heun
        nonlinear tracer substeps, predictor-velocity momentum stages and
        midpoint slow shear for matched tracer faces. Linear and gravity
        splitting errors remain; no whole-model second-order claim is made.
      - process_time_scheme='symmetric_fast_v3' additionally uses two actual
        half-continuity steps around midpoint fast momentum, constrained walls,
        and exact linear bottom drag around the full step. Diffusion, filters
        and coupled forcing errors remain unqualified; not whole-model RK2.
    """
    base = make_fd_params(grid, column_geometry=column_geometry)
    if match_barotropic_transport and (not mode_split or column_geometry != 'nodal_dual_v1'):
        raise ValueError("match_barotropic_transport requires mode_split=True and column_geometry='nodal_dual_v1'")
    if process_time_scheme not in ('legacy', 'consistent_split_v1', 'subcycled_rk2_v2', 'symmetric_fast_v3'):
        raise ValueError("unknown process_time_scheme")
    if process_time_scheme != 'legacy' and not match_barotropic_transport:
        raise ValueError(f"{process_time_scheme} requires match_barotropic_transport=True")
    if column_geometry == 'nodal_dual_v1':
        if not conservative_kv or not localize_conv:
            raise ValueError("nodal_dual_v1 requires conservative_kv=True and localize_conv=True")
    nx, ny, nz = base.nx, base.ny, base.nz
    if projection_niter is None:
        legacy_cap = os.environ.get('OCEAN_PAV_NITER')
        projection_niter_source = 'default' if legacy_cap is None else 'environment:OCEAN_PAV_NITER'
        try:
            projection_niter = 150 if legacy_cap is None else int(legacy_cap)
        except ValueError as error:
            raise ValueError('projection_niter environment value must be an integer') from error
    else:
        projection_niter_source = 'explicit'
    if isinstance(projection_niter, (bool, np.bool_)) or not isinstance(projection_niter, (int, np.integer)) or projection_niter <= 0:
        raise ValueError('projection_niter must be a positive integer')
    if projection_rtol is not None and (not np.isfinite(projection_rtol) or projection_rtol <= 0.):
        raise ValueError('projection_rtol must be finite and positive')
    if projection_preconditioner not in {'none', 'jacobi'}:
        raise ValueError('projection_preconditioner must be none or jacobi')
    if isinstance(projection_max_refinements, (bool, np.bool_)) or not isinstance(projection_max_refinements, (int, np.integer)) or not 0 <= projection_max_refinements <= 2:
        raise ValueError('projection_max_refinements must be an integer from 0 to 2')
    projection_dtype = jnp.float32 if dtype == 'float32' else jnp.float64
    projection_rtol = max(float(projection_rtol or 1e-12), 32. * float(jnp.finfo(projection_dtype).eps))
    if mixed_layer_depth_m is not None:
        if not np.isfinite(mixed_layer_depth_m) or mixed_layer_depth_m < 0.:
            raise ValueError("mixed_layer_depth_m must be finite and nonnegative")
    if mixed_layer_depth_2d is not None:
        depths = np.asarray(mixed_layer_depth_2d)
        if depths.shape != (nx, ny) or not np.all(np.isfinite(depths) & (depths > 0.)):
            raise ValueError("mixed_layer_depth_2d must have grid shape and positive finite depths")
    if dynamic_ice and (not np.isfinite(ice_insulation_scale_m) or ice_insulation_scale_m <= 0.):
        raise ValueError("ice_insulation_scale_m must be finite and positive")

    if forcing is None:
        tau_x_2d = jnp.zeros((nx, ny))
        tau_y_2d = jnp.zeros((nx, ny))
        Q_heat_2d = jnp.zeros((nx, ny))
    else:
        tau_x_2d, tau_y_2d, Q_heat_2d = (jnp.array(f) for f in forcing)

    # Effective shallow-water depth = vertical grid span
    H_sw = float(jnp.sum(jnp.array(grid.dz)))
    dz_norm = (jnp.array(grid.dz).reshape(1, 1, -1) / H_sw)
    if column_geometry == 'nodal_dual_v1':
        column_thickness = np.asarray(base.dz_node) * np.asarray(base.wet_mask_z)
        column_depth = np.sum(column_thickness, axis=-1)
        H_sw = jnp.asarray(np.where(column_depth > 0., column_depth, 1.))
        dz_norm = jnp.asarray(column_thickness) / H_sw[..., None]

    if T_atm is not None and lambda_bulk > 0.0:
        T_atm_3d = jnp.array(T_atm)[:, :, None]
    else:
        T_atm_3d = jnp.zeros((nx, ny, 1))
        lambda_bulk = 0.0

    # ── Surface salinity restoring (Haney) ──
    # dSdt += -(SSS - S_ref)/tau at wet surface cells. tau=0 => off (bit-exact to
    # the pre-restoring solver). S_ref_surf is a (nx, ny) climatological SSS field
    # (e.g. WOA surface salinity); the restoring is a TRUE salt flux (psu/s, no
    # heat_factor -- salinity has no rho*cp). (D24)
    if S_ref_surf is not None and sss_restore_days > 0.0:
        S_ref_2d = jnp.array(S_ref_surf)
        restore_coef_S = 1.0 / (sss_restore_days * 86400.0)
    else:
        S_ref_2d = jnp.zeros((nx, ny))
        restore_coef_S = 0.0

    # ── Diagnostic coastal extra vertical tracer diffusion ──
    if coastal_kappa_v_mask is not None and coastal_kappa_v > 0.0:
        coastal_kappa_v_2d = (
            jnp.array(coastal_kappa_v_mask, dtype=jnp.float64)
            * coastal_kappa_v)
    else:
        coastal_kappa_v_2d = jnp.zeros((nx, ny))

    # ── Diagnostic coastal extra horizontal tracer diffusion ──
    if coastal_kappa_h_mask is not None and coastal_kappa_h > 0.0:
        coastal_kappa_h_2d = (
            jnp.array(coastal_kappa_h_mask, dtype=jnp.float64)
            * coastal_kappa_h)
    else:
        coastal_kappa_h_2d = jnp.zeros((nx, ny))

    # ── Diagnostic coastal extra bulk heat exchange ──
    # Same air-sea form as the bulk flux, but only in the land-adjacent band.
    # This is a more physical alternative to direct SST restoring.
    if coastal_bulk_mask is not None and coastal_bulk_lambda > 0.0:
        coastal_bulk_lambda_2d = (
            jnp.array(coastal_bulk_mask, dtype=jnp.float64)
            * coastal_bulk_lambda)
    else:
        coastal_bulk_lambda_2d = jnp.zeros((nx, ny))

    # ── Diagnostic coastal surface-temperature restoring ──
    # dTdt += -(SST - T_ref)/tau in the land-adjacent band. This is not a
    # physical closure; it is a narrowly scoped attribution experiment for the
    # 0..3-cell cold-bias population.
    if coastal_restore_mask is not None and coastal_restore_days > 0.0 \
            and coastal_restore_T is not None:
        coastal_restore_coef_2d = (
            jnp.array(coastal_restore_mask, dtype=jnp.float64)
            / (coastal_restore_days * 86400.0))
        coastal_restore_T_2d = jnp.array(coastal_restore_T)
    else:
        coastal_restore_coef_2d = jnp.zeros((nx, ny))
        coastal_restore_T_2d = jnp.zeros((nx, ny))

    # ── Lateral sponge (polar-edge Rayleigh damping) ──
    # Applied at both lat edges (the polar cap rows). Cosine-tapered from
    # r_max at the edge to 0 at sponge_cells into the interior.
    if sponge_days > 0.0 and sponge_cells > 0:
        r_max = 1.0 / (sponge_days * 86400.0)
        nc = int(sponge_cells)
        j = np.arange(ny)
        taper = np.zeros(ny)
        edge = np.minimum(j, ny - 1 - j)   # distance to nearest N/S edge
        in_band = edge < nc
        taper[in_band] = 0.5 * (1.0 + np.cos(np.pi * edge[in_band] / nc))
        sponge_2d_np = (r_max * taper).reshape(1, ny)
        sponge_rate_2d = jnp.array(np.broadcast_to(sponge_2d_np, (nx, ny)))
        sponge_rate = sponge_rate_2d[:, :, None]   # (nx, ny, 1)
        # The relaxation target must be the initial state AS THE MODEL SEES IT,
        # i.e. carrying the same land / below-seafloor sentinel that init_state
        # installs (T_ref / S_ref). Relaxing land toward the raw, unmasked T_init
        # drags the land value off that sentinel and breaks the "land and ghost
        # hold their pre-step value" invariant the diffusion mask establishes,
        # re-opening the coastline cliff the pressure integral then reads. (D19)
        wm3 = jnp.asarray(base.wet_mask_z)
        if T_init is not None:
            T_clim_3d = jnp.array(T_init) * wm3 + (1.0 - wm3) * physics.T_ref
            S_clim_3d = (jnp.array(S_init) if S_init is not None
                         else jnp.full_like(T_clim_3d, physics.S_ref))
            S_clim_3d = S_clim_3d * wm3 + (1.0 - wm3) * physics.S_ref
        else:
            # No initial field: init_state fills T_ref / S_ref everywhere, so
            # that is the only target that leaves the sponge a no-op.
            T_clim_3d = jnp.full((nx, ny, nz), physics.T_ref)
            S_clim_3d = jnp.full((nx, ny, nz), physics.S_ref)
    else:
        sponge_rate_2d = jnp.zeros((nx, ny))
        sponge_rate = jnp.zeros((nx, ny, 1))
        T_clim_3d = jnp.zeros((nx, ny, nz))
        S_clim_3d = jnp.zeros((nx, ny, nz))

    # ── Semi-enclosed-sea eta relaxation mask (Mediterranean artifact fix) ──
    # A parameterized lat/lon box; 0 rate = off (bit-exact to the old runs). The
    # buffer band (mask tapers 1 -> 0 over eta_relax_buffer degrees around the box
    # edge) keeps the PGF smooth at the mask boundary so the relaxation itself
    # cannot seed a new PGF cliff at Gibraltar. (D23)
    if eta_relax_days > 0.0 and eta_relax_box is not None:
        lon0, lon1, lat0, lat1 = eta_relax_box
        rate = 1.0 / (eta_relax_days * 86400.0)
        # The global grid lon is in [-180, 180) (WOA convention); a
        # Mediterranean box does not straddle the seam, so plain comparisons
        # suffice (a straddling box would need lon wrap handling — not used).
        lon2d, lat2d = np.meshgrid(np.asarray(grid.lon), np.asarray(grid.lat),
                                   indexing='ij')
        core = ((lon2d >= lon0) & (lon2d <= lon1)
                & (lat2d >= lat0) & (lat2d <= lat1))
        # Box-coordinate distance (degrees) outside the box, for the taper.
        dlon = np.maximum(np.maximum(lon0 - lon2d, lon2d - lon1), 0.0)
        dlat = np.maximum(np.maximum(lat0 - lat2d, lat2d - lat1), 0.0)
        d_out = np.maximum(dlon, dlat)
        buf = float(eta_relax_buffer)
        mask_np = np.where(core, 1.0,
                           np.where(d_out < buf, 0.5 * (1.0 + np.cos(
                               np.pi * d_out / buf)), 0.0))
        mask_np = mask_np * np.asarray(base.wet_mask)   # ocean points only
        eta_relax_mask = jnp.array(mask_np)
        eta_relax_rate = rate
    else:
        eta_relax_mask = jnp.zeros((nx, ny))
        eta_relax_rate = 0.0

    # Nonlinear products (u*du/dx, ...) amplify the 2-dx grid-scale mode. Lon is
    # periodic -> a 2/3 FFT rule in lon is exact; the closed lat wall can't use
    # FFT, so the lat 2-dx is killed by a 5-pt binomial low-pass in _dealias_h_fd.
    # That helper is applied to adv_u/adv_v and to the GM/Redi skew-flux tendency
    # ONLY -- not to w or to the flux-form tracer advection, whose exact
    # telescoping it would destroy. (D7, D15, D25)
    _kmax = nx // 2
    _keep = max(1, int(_kmax * 2 / 3))
    _kidx = np.fft.fftfreq(nx) * nx            # 0..nx/2, -nx/2..-1
    _lon_mask_np = (np.abs(_kidx) <= _keep).astype(np.float64).reshape(nx, 1, 1)
    dealias_lon_mask = jnp.array(np.broadcast_to(_lon_mask_np, (nx, 1, 1)))

    # ── Mode split (baroclinic/barotropic) ──
    # n_subcyc = exact number of barotropic subcycles filling dt: dt_bt_eff =
    # dt/n_subcyc (rounded so each subcycle is a uniform dt_bt). With
    # mode_split=False all fields are inert (n_subcyc=0) and _step_impl takes the
    # bit-exact monolithic path. (D12)
    if mode_split:
        n_subcyc = max(1, int(round(dt / dt_bt)))
        dt_bt_eff = dt / n_subcyc
    else:
        n_subcyc = 0
        dt_bt_eff = 0.0

    # ── Compute dtype ──
    # 'float64' (default) = legacy bit-exact path. 'float32' casts every params
    # array and zeros template to fp32 -- on Ada-class GPUs (FP64:FP32 = 1:64)
    # this is the single biggest kernel-time lever. The runner still reads/writes
    # float64 npz (I/O casts at the boundary). The cast of params happens right
    # after FDPhysParams(...).
    state_dtype = jnp.float32 if dtype == 'float32' else jnp.float64

    # ── nu_h subcycle right-sizing (split L half-steps) ──
    # Legacy (None): n_nu = n_subcyc (24 at dt=3600/dt_bt=150) -- comfortably
    # inside the bound, but ~1.6x more Laplacian pairs than the bound needs.
    # nu_nsub='cfl' right-sizes from the true worst metric point (see
    # nu_nsub_for_2d_cfl). An int is honored verbatim (probe/benchmark). (D12)
    if nu_nsub == 'cfl':
        nu_nsub = nu_nsub_for_2d_cfl(physics.nu_h, dt, grid.dx_2d, grid.dy)
    elif nu_nsub is not None:
        nu_nsub = max(1, int(nu_nsub))

    conv_nsub = max(1, int(np.ceil(
        physics.kappa_conv * dt / (0.4 * float(jnp.min(jnp.array(grid.dz))) ** 2))))
    adv_nsub = max(1, int(np.ceil(dt * 4.0e-3 / (0.5 * float(jnp.min(jnp.array(grid.dz)))))))
    if column_geometry == 'nodal_dual_v1':
        widths = np.asarray(base.dz_node).ravel()
        distances = np.asarray(base.dz_iface).ravel()
        interface_rate = 1. / distances
        row_rate = (np.pad(interface_rate, (1, 0)) + np.pad(interface_rate, (0, 1))) / widths
        conv_nsub = max(1, int(np.ceil(physics.kappa_conv * dt * np.max(row_rate) / 0.4)))
        adv_nsub = max(1, int(np.ceil(dt * 4.0e-3 / (0.5 * np.min(widths)))))
        if physics.nu_v * (dt / 2.) * np.max(row_rate) > 0.4:
            raise ValueError("nodal_dual_v1 vertical momentum diffusion exceeds its explicit CFL margin")
        maximum_kappa_v = physics.kappa_v + float(np.max(np.asarray(coastal_kappa_v_2d)))
        if maximum_kappa_v * (dt / 2.) * np.max(row_rate) > 0.4:
            raise ValueError("nodal_dual_v1 vertical tracer diffusion exceeds its explicit CFL margin")

    params = FDPhysParams(
        dx_2d=base.dx_2d, dy=base.dy, cos_lat=base.cos_lat,
        inv_dx=base.inv_dx, inv_dy=base.inv_dy,
        inv_dx2=base.inv_dx2, inv_dy2=base.inv_dy2,
        f=base.f, wet_mask=base.wet_mask, wet_mask_z=base.wet_mask_z,
        interior_mask_z=base.interior_mask_z,
        dz_denom_interior=base.dz_denom_interior,
        dz_bnd_top=base.dz_bnd_top, dz_bnd_bot=base.dz_bnd_bot,
        d2z_hm=base.d2z_hm, d2z_hp=base.d2z_hp, d2z_denom=base.d2z_denom,
        d2z_h0_top=base.d2z_h0_top, d2z_h0_bot=base.d2z_h0_bot,
        dz_3d=base.dz_3d, dz_surface=base.dz_surface, dz_iface=base.dz_iface,
        dz_node=base.dz_node,
        surface_mask=base.surface_mask, bottom_mask=base.bottom_mask,
        nx=nx, ny=ny, nz=nz,
        nu_h=physics.nu_h, nu_v=physics.nu_v,
        kappa_h=physics.kappa_h, kappa_v=physics.kappa_v,
        kappa_conv=physics.kappa_conv,
        nu_bi=physics.nu_bi, kappa_bi=physics.kappa_bi,
        T_ref=physics.T_ref, S_ref=physics.S_ref,
        r_bot=physics.r_bot, cd=physics.cd,
        bottom_friction=physics.bottom_friction,
        tau_x_2d=tau_x_2d, tau_y_2d=tau_y_2d, Q_heat_2d=Q_heat_2d,
        H_sw=H_sw, dz_norm=dz_norm, dt=dt,
        T_atm_3d=T_atm_3d, lambda_bulk=lambda_bulk,
        S_ref_2d=S_ref_2d, restore_coef_S=restore_coef_S,
        coastal_restore_coef_2d=coastal_restore_coef_2d,
        coastal_bulk_lambda_2d=coastal_bulk_lambda_2d,
        coastal_kappa_h_2d=coastal_kappa_h_2d,
        coastal_kappa_v_2d=coastal_kappa_v_2d,
        coastal_restore_T_2d=coastal_restore_T_2d,
        sponge_rate=sponge_rate, sponge_rate_2d=sponge_rate_2d,
        T_clim_3d=T_clim_3d, S_clim_3d=S_clim_3d,
        polar_cap_rows=int(polar_cap_rows),
        polar_cap_taper=int(polar_cap_taper),
        dealias_lon_mask=dealias_lon_mask,
        kappa_gm=physics.kappa_gm,
        gm_slope_max=physics.gm_slope_max,
        kappa_redi=physics.kappa_redi,
        eta_relax_mask=eta_relax_mask,
        eta_relax_rate=eta_relax_rate,
        mode_split=bool(mode_split),
        dt_bt=dt_bt_eff,
        n_subcyc=n_subcyc,
        conv_nsub=conv_nsub,
        nu_nsub=nu_nsub,
        use_scan=bool(use_scan),
        # Vertical-advection subcycles: keep dt*(|u|/dx + |v|/dy + w/dz) below
        # ~0.5 per substep. The dominant constraint is w/dz in the 5 m surface
        # layer (equatorial upwelling w~3e-3 m/s -> dt*w/dz ~ 2.2 at dt=3600); the
        # horizontal terms (~1e-6/s at |u|~1 m/s) are negligible. Sized from a
        # nominal w_max=4e-3 m/s and a 0.5 target CFL per substep. Monolithic
        # dt=60-300: dt*w/dz <= 0.18 -> adv_nsub=1, path untouched. (D16)
        adv_nsub=adv_nsub,
        freeze_adv_vel=bool(freeze_adv_vel),
        conservative_kv=bool(conservative_kv),
        project_adv_vel=bool(project_adv_vel),
        localize_conv=bool(localize_conv),
        monotone_adv=bool(monotone_adv),
        fct_adv=bool(fct_adv),
        mixed_layer_depth_m=float(mixed_layer_depth_m or 0.0),
        mixed_layer_mask_2d=(jnp.asarray(mixed_layer_mask, dtype=jnp.float64)
                              if mixed_layer_mask is not None
                              else jnp.ones((nx, ny), dtype=jnp.float64)),
        mixed_layer_depth_2d=(jnp.asarray(mixed_layer_depth_2d,
                                           dtype=jnp.float64)
                              if mixed_layer_depth_2d is not None else None),
        ice_freeze_temp_c=float(ice_freeze_temp_c),
        ice_salt_flux=float(ice_salt_flux),
        dynamic_ice=bool(dynamic_ice),
        ice_insulation_scale_m=float(ice_insulation_scale_m),
        projection_niter=int(projection_niter),
        projection_rtol=projection_rtol,
        projection_preconditioner=projection_preconditioner,
        projection_inv_diagonal=None,
        projection_niter_source=projection_niter_source,
        projection_max_refinements=int(projection_max_refinements),
        column_geometry=column_geometry,
        match_barotropic_transport=bool(match_barotropic_transport),
        process_time_scheme=process_time_scheme,
    )

    if projection_preconditioner == 'jacobi':
        diagonal = _column_projection_diagonal(params)
        params = params._replace(projection_inv_diagonal=1. / jnp.where(diagonal > 0., diagonal, 1.))

    # fp32 cast: params was just built in float64 (numpy defaults); when
    # computing in float32 every array field must be cast too, or XLA inserts
    # implicit f64 promotion kernels that silently kill the fp32 speedup.
    # Scalars (int/float/str/None) are passed through unchanged.
    if dtype == 'float32':
        params = params._replace(**{
            f: (v.astype(state_dtype)
                if isinstance(v, jnp.ndarray) and v.dtype == jnp.float64
                else v)
            for f in params._fields for v in [getattr(params, f)]})

    @jax.jit
    def step(state):
        return _step_impl(state, params)

    # ── Dynamic forcing path ──
    # Same compiled graph as step() except the 2D forcing (tau_x, tau_y, Q_heat)
    # is passed as RUNTIME arguments instead of baked constants. params._replace
    # on the namedtuple swaps the three fields for traced placeholders; XLA
    # compiles ONE graph shared by all 12 monthly snapshots (no 12x memory). With
    # dynamic_forcing=False nothing is built (bit-exact to the pre-dynamic path).
    step_dyn = None
    if dynamic_forcing:
        @jax.jit
        def step_dyn(state, tau_x, tau_y, q_heat, T_atm_3d=None):
            updates = {
                'tau_x_2d': tau_x,
                'tau_y_2d': tau_y,
                'Q_heat_2d': q_heat,
            }
            # Optional runtime bulk target lets monthly atmospheric forcing
            # reuse the single compiled dynamic-forcing graph. Existing
            # callers that omit this argument remain bit-compatible.
            if T_atm_3d is not None:
                updates['T_atm_3d'] = T_atm_3d
            return _step_impl(state, params._replace(**updates))

    @jax.jit
    def diagnostics(state):
        rho_prime = _density_anomaly(state.T, state.S, params)
        rho = RHO_0 + rho_prime
        pressure = _compute_hydrostatic_pressure(state, params)
        w = _compute_vertical_velocity(state, params)
        return rho, pressure, w

    @jax.jit
    def terms_fn(state):
        return _tracer_terms(state, params)

    def init_state(T_init=None, S_init=None):
        u = jnp.zeros((nx, ny, nz), dtype=state_dtype)
        v = jnp.zeros((nx, ny, nz), dtype=state_dtype)
        eta = jnp.zeros((nx, ny), dtype=state_dtype)
        if T_init is not None:
            T = jnp.array(T_init, dtype=state_dtype)
            S = jnp.array(S_init, dtype=state_dtype) if S_init is not None \
                else jnp.full_like(T, physics.S_ref)
        else:
            T = jnp.full((nx, ny, nz), physics.T_ref, dtype=state_dtype)
            S = jnp.full((nx, ny, nz), physics.S_ref, dtype=state_dtype)
        # Mask land + ghost water (layers below seafloor): set to a sentinel.
        # wet_mask_z is the TRUE 3D mask (1 where water exists, 0 on land AND
        # below seafloor). This discards WOA-interpolated T in ghost layers
        # before it can enter the pressure integral or advection.
        T = T * params.wet_mask_z + (1.0 - params.wet_mask_z) * physics.T_ref
        S = S * params.wet_mask_z + (1.0 - params.wet_mask_z) * physics.S_ref
        return JaxStateG(u, v, T, S, eta, jnp.zeros_like(eta))

    # Return arity unchanged when dynamic_forcing=False (all existing
    # callers unpack 3-/5-tuples); with dynamic_forcing=True step_dyn is
    # appended as the last element.
    if dynamic_forcing:
        if return_params:
            return step, init_state, diagnostics, params, terms_fn, step_dyn
        return step, init_state, diagnostics, step_dyn
    if return_params:
        return step, init_state, diagnostics, params, terms_fn
    return step, init_state, diagnostics


# ── MMS verification ─────────────────────────────────────────────
# Manufactured-solution checks for the FD operators. 2nd-order FD should
# converge as error ~ dx^2 (~1e-3 at 1 deg). This is the EXPECTED and honest
# FD baseline -- the point is the convergence ORDER, not the absolute error.

def _mms_grid(nx, ny, res, lat_max=75.0):
    """Metric-only grid + FDParams for the MMS operators (no ETOPO needed).

    `dy` must come from the SAME lat span that defines the rows: deriving it
    from the lon resolution instead left the d/dy check 15.4% off, which the
    RMS/max normalization hid inside a generous threshold.
    """
    lat = np.linspace(-lat_max, lat_max, ny)       # avoid poles (cos->0)
    lon = res * (0.5 + np.arange(nx))
    lon_2d, lat_2d = np.meshgrid(lon, lat, indexing='ij')
    cos_lat = np.cos(np.radians(lat))
    z = np.array([0, -10, -30, -60], dtype=np.float64)

    class _G:
        pass

    g = _G()
    g.dx_2d = np.broadcast_to(R_EARTH * np.radians(res) * cos_lat, (nx, ny)).copy()
    g.dy = R_EARTH * np.radians(2.0 * lat_max / (ny - 1))
    g.cos_lat = cos_lat
    g.f = 2.0 * OMEGA * np.sin(np.radians(lat_2d))
    g.wet_mask = np.ones((nx, ny))                 # all-ocean for MMS
    g.wet_mask_3d = g.wet_mask[..., None]
    g.z = z
    g.dz = np.abs(np.diff(z))
    g.nz = len(z)
    g.nx, g.ny = nx, ny
    return make_fd_params(g), lon_2d, lat_2d, cos_lat


def _mms_run():
    """Run MMS checks on a small synthetic global grid (no ETOPO needed).

    Returns a dict of {test_name: (error, passes_threshold)}.
    Uses a 10° grid so the FD truncation is visible, and analytic
    trigonometric fields.
    """
    nx, ny, res = 36, 14, 360.0 / 36   # 10° resolution
    p, lon_2d, lat_2d, cos_lat = _mms_grid(nx, ny, res)
    nz = p.nz

    # Analytic field: u(lon, lat) = sin(m*lon_rad) * cos(n*lat_rad).
    # This is periodic in longitude (exact wrap) and smooth in latitude.
    # The FD operators use physical-space dx = R*cos(lat)*dlon, so du/dx in
    # m/s = (1/dx) * du/dlon_phys. Analytic du/dx = (1/(R*cosφ)) * m*cos(m*lon)*cos(n*lat).
    lon_rad = np.radians(lon_2d)                 # (nx, ny)
    lat_rad = np.radians(lat_2d)
    m = 2.0   # 2 wavelengths around the globe
    n = 2.0   # 2 oscillations in latitude
    u2d = np.sin(m * lon_rad) * np.cos(n * lat_rad)
    u3d = np.broadcast_to(u2d[..., None], (nx, ny, nz)).copy()

    results = {}

    # ── d/dx: du/dx = [m*cos(m*lon)*cos(n*lat)] / (R*cosφ) ──
    du_dx_fd = np.array(_d_dx(jnp.array(u3d), p))[:, :, 0]
    du_dx_an = (m * np.cos(m * lon_rad) * np.cos(n * lat_rad)) / (R_EARTH * cos_lat[None, :])
    err = np.sqrt(np.mean((du_dx_fd - du_dx_an) ** 2)) / (np.abs(du_dx_an).max() + 1e-30)
    results['d_dx_rel_L2'] = (float(err), err < 0.05)

    # ── d/dy: du/dy = -sin(m*lon)*n*sin(n*lat) / R  (dy = R*dlat) ──
    # Measured on INTERIOR rows: the wall rows use the mirror-ghost no-flux BC,
    # which deliberately differs from the infinite-domain analytic derivative
    # (see _mms_convergence). At 10° with 2 lat-oscillations each wavelength is
    # ~6.5 points, so the 2nd-order truncation is ~1.5% here.
    du_dy_fd = np.array(_d_dy(jnp.array(u3d), p))[:, :, 0]
    du_dy_an = (-np.sin(m * lon_rad) * n * np.sin(n * lat_rad)) / R_EARTH
    sl = slice(2, ny - 2)
    err = (np.sqrt(np.mean((du_dy_fd[:, sl] - du_dy_an[:, sl]) ** 2))
           / (np.abs(du_dy_an[:, sl]).max() + 1e-30))
    results['d_dy_rel_L2'] = (float(err), err < 0.03)

    # ── laplacian of a LINEAR-in-y field: u = y_m = R*lat_rad. On the sphere
    # ∇²(R*lat) = -(tanφ/R) (the metric correction term only; ∂²/∂x²=∂²/∂y²=0).
    # Small magnitude -> sanity check that the laplacian isn't producing junk.
    y_m = R_EARTH * lat_rad
    u_lin3d = np.broadcast_to(y_m[..., None], (nx, ny, nz)).copy()
    lap_fd = np.array(_laplacian_h(jnp.array(u_lin3d), p))[:, :, 0]
    lap_scale = np.abs(lap_fd).max()
    results['laplacian_linear_max'] = (float(lap_scale), lap_scale < 1e-4)

    # ── divergence-free check: u = cos(n*lat_rad) (x-independent), v = 0.
    # ∂u/∂x of an x-independent field = 0 on the sphere, so div = 0.
    u_df = np.cos(n * lat_rad)
    v_df = np.zeros_like(u_df)
    u_df3 = np.broadcast_to(u_df[..., None], (nx, ny, nz)).copy()
    v_df3 = np.broadcast_to(v_df[..., None], (nx, ny, nz)).copy()
    div_fd = np.array(_divergence_h(jnp.array(u_df3), jnp.array(v_df3), p))[:, :, 0]
    div_err = np.abs(div_fd).max()
    results['divergence_free_max'] = (float(div_err), div_err < 1e-10)

    return results


def _mms_convergence():
    """Demonstrate 2nd-order convergence of the FD d/dy operator.

    Runs the d/dy MMS at two resolutions (coarse and fine) and checks the
    error drops by ~4x when dx halves (2nd-order: error ∝ dx²). This is the
    decisive FD verification: it proves the operator is correctly 2nd-order,
    not that it's spectrally accurate. The absolute error at 1° is ~1e-3
    (expected, honest FD baseline); the convergence ORDER is what matters.
    """
    def ddy_error(ny):
        nx, res = 36, 360.0 / 36
        p, lon_2d, lat_2d, _ = _mms_grid(nx, ny, res)
        lat_rad = np.radians(lat_2d)
        lon_rad = np.radians(lon_2d)
        n = 2.0
        u2d = np.sin(2.0 * lon_rad) * np.cos(n * lat_rad)
        u3d = np.broadcast_to(u2d[..., None], (nx, ny, p.nz)).copy()
        du_dy_fd = np.array(_d_dy(jnp.array(u3d), p))[:, :, 0]
        du_dy_an = (-np.sin(2.0 * lon_rad) * n * np.sin(n * lat_rad)) / R_EARTH
        # Measure on INTERIOR rows only (exclude the 2 boundary rows each side).
        # The closed-wall no-flux BC (mirror ghost, ∂u/∂n=0) intentionally
        # deviates from the analytic infinite-domain derivative at the wall —
        # that is the correct physics of a closed wall, not an accuracy loss.
        # The interior rows remain true 2nd-order central differences.
        sl = slice(2, ny - 2)
        return np.sqrt(np.mean((du_dy_fd[:, sl] - du_dy_an[:, sl]) ** 2)) / (
            np.abs(du_dy_an[:, sl]).max() + 1e-30)

    e_coarse = ddy_error(ny=14)
    e_fine = ddy_error(ny=28)
    ratio = e_coarse / (e_fine + 1e-30)
    # 2nd-order: halving dx -> error /4, so ratio ~ 4. Accept ratio > 3.
    return {'ddy_coarse': e_coarse, 'ddy_fine': e_fine,
            'convergence_ratio': ratio, 'passes': bool(ratio > 3.0)}


if __name__ == "__main__":
    print("=== Global FD Solver — MMS verification ===")
    res = _mms_run()
    all_pass = True
    for name, (err, ok) in res.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {name:30s} = {err:.3e}  [{status}]")
    print()
    print("--- 2nd-order convergence check (d/dy) ---")
    conv = _mms_convergence()
    print(f"  coarse (ny=14)  d/dy rel L2 = {conv['ddy_coarse']:.3e}")
    print(f"  fine   (ny=28)  d/dy rel L2 = {conv['ddy_fine']:.3e}")
    print(f"  convergence ratio (expect ~4 for 2nd-order) = {conv['convergence_ratio']:.2f}  "
          f"[{'PASS' if conv['passes'] else 'FAIL'}]")
    if not conv['passes']:
        all_pass = False
    print()
    print("ALL PASS" if all_pass else "SOME FAILED")
