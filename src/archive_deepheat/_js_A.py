"""
Global Finite-Difference Ocean Solver — hydrostatic primitive equations, JAX.

Companion to jax_solver.py (regional pseudo-spectral). This solver uses:
  - 2nd-order finite differences on a global 1° lat-lon grid (lon-periodic),
  - spherical metric factors (dx = R*cos(lat)*dlon, varies with latitude),
  - a real wet_mask for no-flux land boundaries,
  - full 2D Coriolis f = 2*Omega*sin(lat).

The spectral regional solver (jax_solver.py) is UNTOUCHED and retained as a
cross-validation baseline. This file is the new FD verification target.

G1 (this file, initial): FD horizontal operators + vertical operators +
MMS (manufactured-solution) verification. The FD operators are pure functions
taking (field, params); params is a lightweight struct carrying the metric
fields. No time integration yet (G2).

Convention (matches regional solver + grid.py):
  - 3D fields: (nx, ny, nz), axis 0 = lon (periodic), axis 1 = lat, axis 2 = z
  - 2D fields: (nx, ny)
  - z negative downward, z=0 at surface
"""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from collections import namedtuple

from config import RHO_0, ALPHA_T, BETA_S, C_P, G_EARTH, R_EARTH, OMEGA


# ── State (same structure as regional solver) ──────────────────────
JaxStateG = namedtuple('JaxStateG', ['u', 'v', 'T', 'S', 'eta'])


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
    'wet_mask_3d',      # (nx, ny, 1) for 3D broadcast (column-uniform)
    'wet_mask_z',       # (nx, ny, nz) TRUE vertical wet mask: 1 where layer is
                        # above the seafloor, 0 below (ghost water excluded).
                        # Used in pressure integration to kill the spurious PGF
                        # at steep topography (ghost-water-column bug fix).
    'interior_mask',    # (nx, ny) 1 in the interior, 0 on the N/S boundary
                        # rows (j=0, j=ny-1). Used to enforce the no-flux wall:
                        # the normal (meridional) velocity v is zeroed here so
                        # no flow crosses the closed N/S truncation wall.
    'interior_mask_z',  # (nx, ny, 1) broadcast of interior_mask for 3D fields.
    # Vertical grid (non-uniform z-levels, same as regional)
    'dz_denom_interior', 'dz_bnd_top', 'dz_bnd_bot',
    'd2z_hm', 'd2z_hp', 'd2z_denom', 'd2z_h0_top', 'd2z_h0_bot',
    'dz_3d', 'dz_surface', 'dz_iface', 'dz_node',
    'surface_mask', 'bottom_mask',   # (1,1,nz)
    # Dimensions
    'nx', 'ny', 'nz',
])


def make_fd_params(grid):
    """Build FDParams from a GlobalOceanGrid (grid.py).

    Precomputes all metric inverse fields so the FD operators are pure
    array arithmetic (no division inside the hot loop).
    """
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
    # True 3D wet mask (layer-resolved): from grid if available, else fall
    # back to column-uniform (regional/synthetic grids without bathymetry).
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

    # Vertical grid coefficients (identical math to regional _compute_params)
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
    # Node-cell thickness: the model is NODE-based (fields live on z-levels),
    # so the "cell" around node k spans halfway to each neighbor:
    # dz_node[0] = |z1-z0|, dz_node[k] = 0.5*(|zk-zk-1|+|zk+1-zk|),
    # dz_node[-1] = |zN-1-zN-2|. Length nz. Used by the interface flux form
    # for tend_v[k] = (F[k-1/2]-F[k+1/2]) / dz_node[k].
    dz_node = jnp.concatenate([
        jnp.array([float(jnp.abs(z[1] - z[0]))]),
        0.5 * (jnp.abs(jnp.diff(z))[:-1] + jnp.abs(jnp.diff(z))[1:]),
        jnp.array([float(jnp.abs(z[-1] - z[-2]))]),
    ]).reshape(1, 1, -1)

    surface_mask = jnp.zeros(nz).at[0].set(1.0).reshape(1, 1, -1)
    bottom_mask = jnp.zeros(nz).at[-1].set(1.0).reshape(1, 1, -1)

    return FDParams(
        dx_2d=dx_2d, dy=dy, cos_lat=cos_lat,
        inv_dx=inv_dx, inv_dy=inv_dy, inv_dx2=inv_dx2, inv_dy2=inv_dy2,
        f=f, wet_mask=wet_mask, wet_mask_3d=wet_mask_3d,
        wet_mask_z=wet_mask_z,
        interior_mask=interior_mask, interior_mask_z=interior_mask_z,
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
    """Zonal derivative du/dx, 2nd-order central FD, longitude-periodic.

    On a lat-lon grid dx varies with latitude, so the inverse spacing is a
    2D field. Axis 0 (lon) wraps via jnp.roll. Land points: the derivative
    is computed everywhere then masked by wet_mask_3d where appropriate by
    the caller (advection applies its own masking).
    """
    return (jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) * (0.5 * p.inv_dx)


def _d_dy(u, p):
    """Meridional derivative du/dy, 2nd-order central FD with NO-FLUX WALLS.

    Axis 1 (lat) does NOT wrap. The N/S domain edges are CLOSED walls: the
    one-sided extrapolation stencil (-3u0+4u1-u2) it replaced is unbounded
    for advection and drove the tracer blow-up (T→thousands at the boundary
    row). The no-flux wall uses a mirror ghost cell (ghost = boundary value,
    `mode='edge'` reflection about the wall face at the cell edge) so the
    boundary row gets a STABLE central difference with zero normal flux:

        du/dy|_0 = (u1 - u_ghost)/(2 dy) = (u1 - u0)/(2 dy)

    This enforces ∂u/∂n = 0 at the wall face (no diffusive flux) while the
    normal-velocity mask (v=0 at boundary rows, enforced in _step_impl +
    _free_surface_step_fd) kills the advective flux. Together = closed wall.
    """
    # Mirror-pad the lat axis by 1 cell each side (ghost = boundary value).
    # ndim-agnostic: pad only axis 1.
    pad = [(0, 0)] * u.ndim
    pad[1] = (1, 1)
    u_pad = jnp.pad(u, pad, mode='edge')
    return (u_pad[:, 2:] - u_pad[:, :-2]) * (0.5 * p.inv_dy)


def _divergence_conservative(ubt, vbt, p):
    """Mass-conserving horizontal divergence of the barotropic velocity on the
    sphere.

    Flux form whose face fluxes are zeroed at every wet/dry interface (a face
    is 'open' only when BOTH adjacent cells are wet) AND at the N/S truncation
    walls (closed basin). On the lat-lon grid the cell area A_ij = dx*dy
    = R^2*cos(lat)*dphi^2 varies with latitude, so mass = sum(eta*A_ij) is an
    AREA-WEIGHTED sum. The divergence must therefore telescope to zero under
    the AREA-WEIGHTED inner product, not the unweighted one.

    Zonal term: div_x = (uface_+ - uface_-) * inv_dx_i, with the per-cell
    inv_dx_i = 1/(R*cos_i*dphi) = dy/A_i. Because the face difference is
    weighted by 1/A_i, sum_i A_i*div_x telescopes exactly (verified:
    machine-zero residual).

    Meridional term: the SCALAR inv_dy = 1/(R*dphi) does NOT carry the
    cos(lat) factor, so sum_j A_j*div_y does NOT telescope — a per-step mass
    leak (~1.7e-7 m^3 per unit velocity) that grows exponentially and drives
    the eta drift to the 15 m watchdog by day 9. The correct spherical form is

        div_y = (1/cos_j) * d(v*cos)/dy

    i.e. the face MASS flux is v*cos(face_lat), differenced and divided by the
    cell cos_j. With cos_face = 0.5*(cos_j + cos_{j+1}) this telescopes under
    the area weight (verified: machine-zero, vs 1.7e-7 leak for the old form).

    Args:
      ubt, vbt: (nx, ny) 2D barotropic velocity, already masked to wet cells.
    Returns:
      (nx, ny) divergence.
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
    vface_open = vface_open.at[:, -1].set(0.0)        # south wall closed
    vface = 0.5 * (vbt + jnp.roll(vbt, -1, axis=1)) * vface_open
    cos_face = 0.5 * (cos_lat + jnp.roll(cos_lat, -1))   # cos at face j+1/2
    fcos = vface * cos_face[None, :]                     # mass flux v*cos(face)
    fcos_in = jnp.roll(fcos, 1, axis=1)                  # flux at face j-1/2
    fcos_in = fcos_in.at[:, 0].set(0.0)                  # north wall closed
    div_y = (fcos - fcos_in) * p.inv_dy / cos_lat[None, :]
    return div_x + div_y


def _gradient_conservative(eta, p):
    """Exact discrete adjoint of _divergence_conservative, under the
    cell-area-weighted inner product <a,b> = Sum(a*b*A_ij), A_ij = dx_ij*dy.

    This makes the forward-backward free-surface pair ENERGY-NEUTRAL: the
    PE/KE cross terms cancel via discrete summation by parts
        <u, grad eta>_A = -<eta, div u>_A
    so a free gravity wave neither grows nor decays (|lambda|=1) on the masked,
    non-uniform lat-lon grid. The centered _d_dx/_d_dy gradient is NOT this
    adjoint (it ignores the open-face gating AND, on the non-uniform grid, the
    per-cell inv_dx weighting), so pairing it with the conservative divergence
    injects energy at basin scale (pure free wave: E grows ~22x/40steps).

    Derivation (zonal, axis 0 periodic): the conservative divergence is
        div_x_i = (Fx_{i+1/2} - Fx_{i-1/2}) / A_i
    with face flux Fx_{i+1/2} = 0.5*(u_i+u_{i+1})*open_{i+1/2}*dy. Summation by
    parts gives the adjoint
        grad_x_i = inv_dx_i * [0.5*open_{i+1/2}*(eta_{i+1}-eta_i)
                             + 0.5*open_{i-1/2}*(eta_i-eta_{i-1})]
    (the FACE DIFFERENCE, not the face average — the average was the earlier
    wrong guess). inv_dx_i = 1/dx_i = dy/A_i lives at the cell, consistent with
    the divergence's per-cell inv_dx. Meridional axis is analogous with closed
    walls (boundary faces open=0).

    Args:
      eta: (nx, ny) 2D sea-surface height, already masked to wet cells.
    Returns:
      (grad_x, grad_y) each (nx, ny).
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
    # Meridional (axis 1, closed walls): adjoint of the spherical divergence.
    # div_y_j = inv_dy/cos_j * (cos_{j+1/2}*0.5*o+*(v_j+v_{j+1})
    #                           - cos_{j-1/2}*0.5*o-*(v_{j-1}+v_j))
    # Under <eta,div>_A = sum eta_j*div_j*A_j with A_j=cos_j*dx*dy, the cos_j
    # cancels, and summation by parts gives the adjoint (face-difference form,
    # weighted by the face cos and divided by the cell cos):
    #   grad_y_j = inv_dy/cos_j * 0.5*[ cos_{j+1/2}*o+*(eta_{j+1}-eta_j)
    #                                 + cos_{j-1/2}*o-*(eta_j-eta_{j-1}) ]
    open_yp = wm * jnp.roll(wm, -1, axis=1)
    open_ym = wm * jnp.roll(wm, 1, axis=1)
    # Close the two boundary faces (adjoint of the closed truncation walls).
    open_yp = open_yp.at[:, -1].set(0.0)               # south wall
    open_ym = open_ym.at[:, 0].set(0.0)                # north wall
    cos_face_p = 0.5 * (cos_lat + jnp.roll(cos_lat, -1))   # face j+1/2
    cos_face_m = 0.5 * (cos_lat + jnp.roll(cos_lat, 1))    # face j-1/2
    d_eta_yp = (jnp.roll(eta, -1, axis=1) - eta) * open_yp * cos_face_p[None, :]
    d_eta_ym = (eta - jnp.roll(eta, 1, axis=1)) * open_ym * cos_face_m[None, :]
    grad_y = (p.inv_dy / cos_lat[None, :]) * 0.5 * (d_eta_yp + d_eta_ym)
    return grad_x, grad_y


def _gradient_conservative_3d(field, p):
    """3D extension of _gradient_conservative: the masked, cos(lat)-weighted
    adjoint gradient applied per-layer to a (nx, ny, nz) field.

    Same face-gating (open iff BOTH adjacent cells wet, via wet_mask_z) and
    cos(lat) face weighting as the 2D version, so the horizontal gradient of
    a 3D pressure field does not reach into land zeros at coastlines (the
    bare centered _d_dx/_d_dy does, injecting energy there). Used for the 3D
    hydrostatic pressure-gradient force so it is consistent with the
    barotropic conservative divergence / gradient pair.
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
    open_yp = open_yp.at[:, -1, :].set(0.0)           # south wall
    open_ym = open_ym.at[:, 0, :].set(0.0)            # north wall
    cos_face_p = 0.5 * (cos_lat + jnp.roll(cos_lat, -1))   # face j+1/2
    cos_face_m = 0.5 * (cos_lat + jnp.roll(cos_lat, 1))    # face j-1/2
    shp = [1, p.ny, 1]
    d_yp = (jnp.roll(field, -1, axis=1) - field) * open_yp * cos_face_p.reshape(shp)
    d_ym = (field - jnp.roll(field, 1, axis=1)) * open_ym * cos_face_m.reshape(shp)
    grad_y = (p.inv_dy / cos_lat[None, :, None]) * 0.5 * (d_yp + d_ym)
    return grad_x, grad_y


def _divergence_conservative_3d(Fx, Fy, p):
    """3D mass-conserving horizontal divergence of a tracer flux, the exact
    adjoint of _gradient_conservative_3d under the area-weighted inner product.

    Face fluxes are zeroed at every wet/dry interface (open iff BOTH adjacent
    cells wet, via wet_mask_z) and at the N/S truncation walls, so no flux
    crosses a coastline or the basin boundary. This is the conservative pair
    to _gradient_conservative_3d: used for the isopycnal skew-flux divergence
    so the GM/Redi closure neither creates nor destroys integrated tracer at
    land boundaries (the bare _d_dx/_d_dy divergence injects a spurious
    coastal source that drove max|T| from 30 to 47 C in 4 days).

    Args:
      Fx, Fy: (nx, ny, nz) zonal / meridional flux components, already masked.
    Returns:
      (nx, ny, nz) horizontal divergence dFx/dx + dFy/dy.
    """
    wm = p.wet_mask_z                                 # (nx, ny, nz)
    cos_lat = p.cos_lat                               # (ny,)
    # Zonal (axis 0, periodic): face (i+1/2,j,k) open iff cell i and i+1 wet.
    uface_open = wm * jnp.roll(wm, -1, axis=0)
    uface = 0.5 * (Fx + jnp.roll(Fx, -1, axis=0)) * uface_open
    div_x = (uface - jnp.roll(uface, 1, axis=0)) * p.inv_dx[..., 0:1]
    # Meridional (axis 1, closed walls): face (i,j+1/2,k) open iff j,j+1 wet.
    vface_open = wm * jnp.roll(wm, -1, axis=1)
    vface_open = vface_open.at[:, -1, :].set(0.0)    # south wall closed
    vface = 0.5 * (Fy + jnp.roll(Fy, -1, axis=1)) * vface_open
    cos_face = 0.5 * (cos_lat + jnp.roll(cos_lat, -1))   # cos at face j+1/2
    shp = [1, p.ny, 1]
    fcos = vface * cos_face.reshape(shp)                  # mass flux v*cos(face)
    fcos_in = jnp.roll(fcos, 1, axis=1)
    fcos_in = fcos_in.at[:, 0, :].set(0.0)                 # north wall closed
    div_y = (fcos - fcos_in) * p.inv_dy / cos_lat[None, :, None]
    return div_x + div_y


def _laplacian_h(u, p):
    """Horizontal Laplacian on the sphere:

        ∇²u = 1/cosφ ∂/∂φ(cosφ ∂u/∂φ) + 1/cos²φ ∂²u/∂λ²
            = ∂²u/∂y² - (tanφ/R) ∂u/∂y + (1/cos²φ) ∂²u/∂x²

    where ∂/∂y = (1/R)∂/∂φ, ∂/∂x = 1/(R cosφ)∂/∂λ.
    Implemented in physical (meter) space: ∂²/∂x² and ∂²/∂y² via FD, plus
    the spherical correction term -(tanφ/R)·∂u/∂y = -(sinφ/(R cosφ))·∂u/∂y.
    """
    # ∂²u/∂x²: FACE-GATED conservative form. The bare central stencil reads
    # the wet/ghost mask step directly: ghost nodes hold the T_ref sentinel
    # (+15 C vs ~+1 C real deep water), so the inner Laplacian spikes
    # (+14 K/dx² at every wet/ghost face) and the OUTER Laplacian of that
    # spike is a huge dipole — at kappa_bi=2e14 (L-step, invisible to
    # terms_fn) this warms the wet coastal nodes toward +15 at ~+0.3 K/d
    # (gpu365_fgate d150 k13: T[x=138] already drifted 1.2 -> 12.1 C) while
    # advection of the resulting zonal gradient cools the interior at up to
    # -13 K/d at d365 -> FAIL_DRIFT max|T|=220. Gating the face differences
    # (zero across any wet/ghost or wet/dry face, open iff both cells wet)
    # makes the Laplacian see a flat profile across closed faces — the
    # physical no-flux BC — and kills the halo at the source.
    wm = p.wet_mask_z                                   # (nx, ny, nz)
    d2u_dx2 = ((jnp.roll(u, -1, axis=0) - u) * wm * jnp.roll(wm, -1, axis=0)
               - (u - jnp.roll(u, 1, axis=0)) * wm * jnp.roll(wm, 1, axis=0)) * p.inv_dx2
    # ∂²u/∂y²: central second FD with NO-FLUX WALL (mirror ghost cell),
    # PLUS the same wet/wet face gate in y. Same `mode='edge'` reflection as
    # _d_dy: ghost = boundary value, giving zero normal gradient at the wall
    # face and a stable central 2nd diff:
    #   d²u/dy²|_0 = (u1 - 2u0 + u_ghost)/dy² = (u1 - u0)/dy²
    # The prior one-sided (-2,-5,4,-1) stencil extrapolated and amplified the
    # grid-scale mode that blew up the tracer at the boundary row.
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


def _biharmonic_h(u, p):
    """Biharmonic ∇⁴u = ∇²(∇²u). Composed from two Laplacian applications.

    Both applications use the face-gated _laplacian_h, so the mask-step
    artifacts (ghost sentinel cliff) are excluded from ∇²u itself: no halo
    at wet/ghost boundaries, and the ghost-side result of the inner Laplacian
    (irrelevant, masked away downstream) cannot seed the outer one.
    """
    return _laplacian_h(_laplacian_h(u, p), p)


def _laplacian_h_2d(u, v, p):
    """Horizontal Laplacian on a pair of 2D fields (mode-split utilities).

    Currently UNUSED: an earlier design ran nu_h dissipation inside the
    barotropic subcycle on the depth-mean velocity, but the baroclinic SHEAR
    (the main thing nu_h must damp) is invisible to the depth mean — the
    real-grid split run then grew +0.65 m/s/step at a trench node and blew up
    via tracer advection. nu_h now runs subcycled in the L half-steps on the
    full 3D field (see _linear_half_step). Kept as a small utility; delete if
    it finds no other user.

    Same face-gated construction as _laplacian_h (wet/wet face differences in
    x; mirror-ghost + gate in y; spherical -(tanφ/R)∂/∂y correction), applied
    to the 2D (nx, ny) fields. Returns (lap_u, lap_v).
    """
    wm = p.wet_mask                                     # (nx, ny)
    inv_dx2 = p.inv_dx2[:, :, 0] if p.inv_dx2.ndim == 3 else p.inv_dx2
    d2u_dx2 = ((jnp.roll(u, -1, axis=0) - u) * wm * jnp.roll(wm, -1, axis=0)
               - (u - jnp.roll(u, 1, axis=0)) * wm * jnp.roll(wm, 1, axis=0)) * inv_dx2
    pad = [(0, 0), (1, 1)]
    u_pad = jnp.pad(u, pad, mode='edge')
    wm_pad = jnp.pad(wm, pad, mode='edge')
    d2u_dy2 = ((u_pad[:, 2:] - u_pad[:, 1:-1]) * wm_pad[:, 2:] * wm_pad[:, 1:-1]
               - (u_pad[:, 1:-1] - u_pad[:, :-2]) * wm_pad[:, 1:-1] * wm_pad[:, :-2]) * p.inv_dy2
    du_dy = _d_dy(u[:, :, None], p)[:, :, 0]
    sin_lat_row = p.f[0, :] / (2.0 * OMEGA)
    tan_over_R = (sin_lat_row / p.cos_lat) / R_EARTH
    corr = -du_dy * tan_over_R[None, :]
    lap_u = d2u_dx2 + d2u_dy2 + corr

    d2v_dx2 = ((jnp.roll(v, -1, axis=0) - v) * wm * jnp.roll(wm, -1, axis=0)
               - (v - jnp.roll(v, 1, axis=0)) * wm * jnp.roll(wm, 1, axis=0)) * inv_dx2
    v_pad = jnp.pad(v, pad, mode='edge')
    d2v_dy2 = ((v_pad[:, 2:] - v_pad[:, 1:-1]) * wm_pad[:, 2:] * wm_pad[:, 1:-1]
               - (v_pad[:, 1:-1] - v_pad[:, :-2]) * wm_pad[:, 1:-1] * wm_pad[:, :-2]) * p.inv_dy2
    dv_dy = _d_dy(v[:, :, None], p)[:, :, 0]
    corr_v = -dv_dy * tan_over_R[None, :]
    lap_v = d2v_dx2 + d2v_dy2 + corr_v
    return lap_u, lap_v


def _divergence_h(u, v, p):
    """Horizontal divergence du/dx + dv/dy from FACE-FLUX differences.

    div_x = (Fx[i] - Fx[i-1])*inv_dx with the SAME face velocity
    Fx = 0.5*(u + u[i+1]) * wet/wet gate that _advection_scalar's flux-form
    budget uses. On fully-wet nodes this is algebraically identical to the
    2nd-order central difference 0.5*(u[i+1]-u[i-1])*inv_dx (the face
    averages telescope), but at coastal-wall nodes the two disagree
    radically: the central difference measures the node's SHEAR
    (0.5*(u_east - 0) with the land neighbor's u=0), while the face-flux
    form measures the node's real INFLOW (0.5*(u_node + u_east), the actual
    flux through its open face). With the central form the w diagnosis and
    the tracer budget saw different velocity fields at every wall: at the
    Peru-corner node (313.5E, 0.5S, k=11) a ~0.28 m/s depth-uniform inflow
    entered the tracer budget through the open east face while the
    diagnosed w saw ~zero local divergence -> no vertical branch was
    diagnosed to close the budget, and S piled +0.7 PSU/step (38.6 -> 41.3
    by step 8, 68 by step 30). The face-flux form makes w integrate exactly
    the convergence the advection sees: column convergence -> w grows
    upward through the column to w[0] = -∫div dz = the rigid-lid leak =
    the eta tendency the barotropic subcycle is simultaneously absorbing,
    and the Edit-2 Fz_top closure passes it through the surface node.

    Meridional: lat is NOT periodic — mirror-pad the axis (same edge-pad
    wall form as _d_dy / _laplacian_h) so the N/S truncation walls are
    closed (ghost = boundary value, v masked to 0 there anyway), with the
    same cos(face_lat) factor on the flux and 1/cos(cell lat) on the
    divergence as _advection_scalar / _divergence_conservative.
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


def _vertical_transport_iface(u, v, p):
    """Interface volume transports Fz (nx, ny, nz+1), DOWNWARD-positive.

    Built by the EXACT discrete inverse of _divergence_h (the same face
    fluxes _advection_scalar's budget uses): Fz[nz] = 0 (seafloor) and

        (Fz[k+1] - Fz[k]) / dz_node[k] = -div_h[k]   for every layer k,

    so Fz[k] = sum_{m>=k} div_h[m] * dz[m]. Feeding Fz to the donor-cell
    vertical flux makes div_x + div_y + div_z cancel POINTWISE for a uniform
    tracer — the flux-form operator is then exactly (not approximately)
    conservative and neutrally transportive. The previous node-w path
    (trapezoid div_avg + cumsum + _dealias_h_fd) could not invert the
    discrete divergence: the leftover residual reached 1.2e-5/s, and
    resid * T is a T-PROPORTIONAL source — a 4%/step exponential pump
    (measured resid*S = 4.4e-4 PSU/s, max|T| 29.6 -> 52.6 over 30 steps at
    the North Brazil Current node) that no CFL limit can remove because it
    is not a transport term at all.

    Fz[..., 0] is the column-integrated horizontal divergence — the
    rigid-lid leak the barotropic subcycle simultaneously absorbs as the
    eta tendency; the scalar advection passes it through the surface node
    as Fz_top = Fz[0] * T[0] (MOM-style rigid-lid closure).
    """
    div_h = _divergence_h(u, v, p)
    # div_h is per-LAYER (nz entries) — weight by the per-node cell
    # thickness dz_node (1,1,nz), NOT dz_3d: dz_3d is the interface-centered
    # (1,1,nz-1) spacing array (z[k]-z[k+1] between nodes) and cannot
    # multiply the per-layer divergence.
    integrand = div_h * p.dz_node                    # (nx, ny, nz)
    # cumsum from the bottom upward: Fz_int[k] = sum_{m=k..nz-1} integrand[m]
    Fz_int = jnp.cumsum(integrand[..., ::-1], axis=-1)[..., ::-1]
    return jnp.concatenate([Fz_int, jnp.zeros_like(Fz_int[..., :1])], axis=-1)


# ── Vertical operators (identical to regional solver) ──────────────

def _fill_ghost_bottom(u, p):
    """Seafloor no-flux fill: extend each column's bottom wet value downward.

    Ghost layers (below the seafloor, wet_mask_z==0) hold T_ref/S_ref from
    init_state and never evolve. Any vertical stencil that reaches them
    (centered _d_dz spans k-1..k+1) mixes the 15 C ghost reservoir into the
    bottom wet layer's derivative — at sills (bottom wet layer k=12) that is
    a spurious ~0.08 K/day seafloor heat flux in the GM/Redi closure, and a
    NEGATIVE (inverted) drho_dz that pins the isopycnal slope at its clip.
    Replicating the bottom wet value into the ghost layers is the standard
    no-flux seafloor treatment (MOM/NEMO fill ghost cells with the bottom
    value): stencils then see a zero gradient across the floor and the
    bottom-wet one-sided derivative gets the correct stable sign.
    Only the closure operators apply this; the background diffusion/conv
    BC is left as-is (their ghost pull is ~0.006 K/day, a separate issue).
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

    Boundary nodes use the zero-flux (ghost-point) form 2*(C1 - C0)/h0^2:
    the ghost value Cg = C1 (mirror reflection through the boundary node)
    makes the centered curvature (C1 - 2*C0 + Cg)/h0^2 diffusive at the
    boundary. The previous form (C2 - 2*C1 + C0)/h0^2 is the centered
    curvature AT node 1 applied as the tendency of node 0 -- i.e.
    anti-diffusive there: it pushed a boundary anomaly AWAY from the
    interior value. With kappa_conv=0.05 that feedback amplified initial
    WOA salty-over-fresh surface profiles into the ITCZ salinity runaway
    that NaN'd the 365d run at day 135 (conv_S = +84.5 PSU/day at the
    worst cell; the zero-flux form gives -89.9 PSU/day, clearing the
    instability). Shared by conv/diff_v/momentum-vdiff, all of which
    want the same no-flux boundary condition.
    """
    d2u_interior = (
        u[..., 2:] * p.d2z_hm + u[..., :-2] * p.d2z_hp
        - u[..., 1:-1] * (p.d2z_hm + p.d2z_hp)
    ) / p.d2z_denom
    d2u_top = 2.0 * (u[..., 1:2] - u[..., 0:1]) / (p.d2z_h0_top ** 2)
    d2u_bot = 2.0 * (u[..., -2:-1] - u[..., -1:]) / (p.d2z_h0_bot ** 2)
    return jnp.concatenate([d2u_top, d2u_interior, d2u_bot], axis=-1)



def _diff_v_flux_tendency(tracer, kappa, p):
    """Conservative interface-flux vertical diffusion (zero flux at top/bottom).

    Replaces kappa*_d2_dz2, whose node-form stencil is not column-conservative
    on the non-uniform grid (measured: -56.89 ZJ/yr lost at kappa_v=1e-5 on the
    10-yr checkpoint). Same discretization as _conv_flux_tendency.
    """
    if kappa <= 0.0:
        return jnp.zeros_like(tracer)
    tr = _fill_ghost_bottom(tracer, p)
    Fz_i = -kappa * (tr[..., 1:] - tr[..., :-1]) / p.dz_iface
    wet_if = (p.wet_mask_z[..., :-1] > 0.5) & (p.wet_mask_z[..., 1:] > 0.5)
    Fz_i = jnp.where(wet_if, Fz_i, 0.0)
    up = jnp.concatenate([jnp.zeros_like(Fz_i[..., :1]), Fz_i], axis=-1)
    dn = jnp.concatenate([Fz_i, jnp.zeros_like(Fz_i[..., :1])], axis=-1)
    return (up - dn) / p.dz_node * p.wet_mask_z


def _conv_flux_tendency(tracer, conv_mask_3d, kappa, p):
    """Convective mixing in INTERFACE-flux form (exactly column-conservative).

    The node-form implementation it replaces (kappa_conv * mask * _d2_dz2)
    is NOT heat-conservative on the non-uniform grid: the node-form 3-point
    stencil integrated with dz_node weights does not telescope (quadratic-T
    unit test: column integral -6.0e-3 != 0; linear T conserves only by
    accident). Gated by the time-varying full-column conv mask, every
    convective episode therefore created net heat: +0.00025 K/day global
    mean, polar-concentrated (+0.09 C/yr volume mean), which homogenized
    and warmed the polar water columns +1.13 C over the 10-yr run (cap
    2000 m +7.4 C) with an impossible implied +200 W/m2 surface flux.

    This form reuses the GM/Redi vertical discretization (interface fluxes
    F[k+1/2] between nodes k and k+1, tendency (F[k-1/2]-F[k+1/2])/dz_node,
    zero flux at the material top/bottom): the column sum
    sum_k(tend*dz_node) = F[bot]-F[top] = 0 EXACTLY for ANY mask, so
    convection can only redistribute tracer within a column, never create
    it. Also negative-semidefinite (one-sided interface differences), so
    no sawtooth/anti-diffusive modes. The top/bottom-node effective rate is
    half the old mirror form's 2(C1-C0)/h0^2 — same no-flux BC, standard
    FV discretization; the convective adjustment timescale changes only at
    the surface/bottom nodes.

    Args:
        tracer: (nx, ny, nz) T or S.
        conv_mask_3d: (nx, ny, 1) column gate (True where the column convects).
        kappa: [m^2/s] convective diffusivity (p.kappa_conv).
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
    Fz_i = jnp.where(gate & wet_iface_f, Fz_i, 0.0)
    # Padded flux array carries the zero-flux BC at surface/seafloor;
    # tend[k] = (F[k-1/2] - F[k+1/2]) / dz_node[k] (see _redi_skew_flux_tendency).
    up = jnp.concatenate([jnp.zeros_like(Fz_i[..., :1]), Fz_i], axis=-1)
    dn = jnp.concatenate([Fz_i, jnp.zeros_like(Fz_i[..., :1])], axis=-1)
    return (up - dn) / p.dz_node * p.wet_mask_z


# ── FD solver parameters (full, with physics + forcing) ───────────
# Extends FDParams (metric+grid) with the physics constants and forcing
# fields needed for the time integration (G2). Built by make_solver_global.

FDPhysParams = namedtuple('FDPhysParams', [
    # metric + grid (from FDParams)
    'dx_2d', 'dy', 'cos_lat', 'inv_dx', 'inv_dy', 'inv_dx2', 'inv_dy2',
    'f', 'wet_mask', 'wet_mask_3d', 'wet_mask_z',
    'interior_mask', 'interior_mask_z',
    'dz_denom_interior', 'dz_bnd_top', 'dz_bnd_bot',
    'd2z_hm', 'd2z_hp', 'd2z_denom', 'd2z_h0_top', 'd2z_h0_bot',
    'dz_3d', 'dz_surface', 'dz_iface', 'dz_node', 'surface_mask', 'bottom_mask',
    'nx', 'ny', 'nz',
    # physics
    'nu_h', 'nu_v', 'kappa_h', 'kappa_v', 'kappa_conv',
    'nu_bi', 'kappa_bi',
    'T_ref', 'S_ref', 'eos_type', 'r_bot', 'cd', 'bottom_friction',
    # forcing (2D physical-space; no FFT pre-compute in the FD solver)
    'tau_x_2d', 'tau_y_2d', 'Q_heat_2d',
    # free surface
    'H_sw', 'dz_norm', 'dt',
    # bulk air-sea heat flux
    'T_atm_3d', 'lambda_bulk',
    # surface salinity restoring (Haney): relax SSS toward S_clim_surf with
    # an equivalent salt flux. Mirrors the T_atm/lambda_bulk plumbing:
    # dSdt += -restore_coef_S * (S_surf - S_ref_2d) * surface_mask.
    # Physically: an "atmosphere" that supplies/absorbs whatever freshwater
    # flux keeps SSS at climatology (E-P bias correction), replacing the
    # v0.1 solver's missing surface salinity BC (SSS drifted -0.9 psu/kyr).
    'S_ref_2d', 'restore_coef_S',
    # lateral sponge (polar-edge Rayleigh damping; global analogue of the
    # regional N/S-boundary sponge — absorbs wind-driven barotropic energy
    # that Laplacian dissipation can't within its CFL cap)
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
    # Mode split (baroclinic/barotropic). With mode_split=True the free
    # surface no longer runs inside the Strang linear half-step at the
    # baroclinic dt; instead _step_impl ends with n_subcyc barotropic
    # forward-backward subcycles of dt_bt each (exact fill of dt), with the
    # density-PGF coupling held fixed over the baroclinic step (MOM-style
    # forcing lag). This lifts the external-gravity-wave CFL from the
    # baroclinic dt (dt=3600 s at 1 deg is then safe) and gives ~10x
    # wall-clock.
    'mode_split',        # bool: barotropic-subcycle free surface
    'dt_bt',             # s: ACTUAL subcycle dt (= dt / n_subcyc, exact fill)
    'n_subcyc',          # int: barotropic subcycles per baroclinic step
    'conv_nsub',         # int: convective-adjustment subcycles per bc step
    'adv_nsub',          # int: vertical-advection subcycles per bc step
    # nu_h subcycle count in the split L half-steps. None (default) = legacy
    # behavior (n_nu = n_subcyc, 24 substeps at dt=3600/dt_bt=150 — CFL LHS
    # 0.136, a 3.7x margin under the 0.5 FTCS bound). An int right-sizes it
    # from the actual metric (see make_solver_global nu_nsub arg): ~6
    # substeps at the same margin — 4x fewer laplacian pairs per half-step.
    'nu_nsub',
    # True: run the barotropic subcycle as a jax.lax.scan (fused device loop,
    # body compiled once) instead of an unrolled Python loop. Numerically
    # identical; cuts XLA graph size and host launch overhead.
    'use_scan',
])

# Keyword-constructed callers that predate nu_nsub/use_scan (archive_diag
# probes) get the legacy behavior instead of a TypeError.
FDPhysParams.__new__.__defaults__ = (None, False)


# ── EOS (shared with spectral solver; copied to avoid import cycle) ─

def _density_anomaly(T, S, p):
    """rho' = rho - rho_0. Linear EOS branch (global default)."""
    if p.eos_type == 'unesco':
        # UNESCO not needed for global default; fall back to linear.
        return RHO_0 * (-ALPHA_T * (T - p.T_ref) + BETA_S * (S - p.S_ref))
    return RHO_0 * (-ALPHA_T * (T - p.T_ref) + BETA_S * (S - p.S_ref))


def _compute_hydrostatic_pressure(state, p):
    """Full hydrostatic pressure via cumulative trapezoidal integration.

    Ghost-water fix: the density anomaly is masked by wet_mask_z (zeroed
    below the seafloor) BEFORE integration. Layers below the seafloor then
    contribute dp=0, so the cumulative pressure stays constant beneath the
    bottom (no spurious horizontal gradient from columns of different
    ghost-water length). This is the fix for the blow-up at steep topography.
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

    Uses _gradient_conservative_3d (masked, cos(lat)-weighted adjoint) — NOT
    the bare centered _d_dx/_d_dy — so the 3D PGF does not reach into land
    zeros at coastlines. The bare centered gradient injects baroclinic energy
    at every coastline point (it sees land p=0 as a huge pressure drop),
    which advects into the interior and seeds the day-50 equatorial-Atlantic
    blowup even after the barotropic rho-PGF was fixed (the barotropic fix
    only delayed it 5 days). This 3D fix closes the remaining injection path.
    """
    pressure = _compute_hydrostatic_pressure(state, p)
    pgf_x, pgf_y = _gradient_conservative_3d(pressure, p)
    return -pgf_x / RHO_0, -pgf_y / RHO_0


def _barotropic_velocity(u, v, p):
    """Depth-averaged (barotropic) horizontal velocity (same as spectral)."""
    u_avg = 0.5 * (u[..., :-1] + u[..., 1:])
    v_avg = 0.5 * (v[..., :-1] + v[..., 1:])
    ubt = jnp.sum(u_avg * p.dz_norm, axis=-1)
    vbt = jnp.sum(v_avg * p.dz_norm, axis=-1)
    return ubt, vbt


def _compute_bt_rho_pgf(state, p):
    """Barotropic (depth-averaged) PGF from density anomalies (FD).

    Transport-consistent form: the depth average over the WET column of the
    SAME face-gated 3D baroclinic PGF that the 3D momentum feels
    (_gradient_conservative_3d — the masked, cos(lat)-weighted adjoint of the
    3D divergence, same construction as _compute_pressure_gradient), normalized
    by H_sw to match ubt = transport / H_sw in _barotropic_velocity (u = 0 in
    ghost water):

        F_rho = (1/H_sw) * SUM_k pgf3d_layer_k * dz_k * wet_iface_k

    Uses _gradient_conservative_3d, NOT the bare centered _d_dx/_d_dy, so the
    rho-PGF cannot reach into land zeros at coastlines (see the day-45
    Gulf-of-Guinea note in _compute_pressure_gradient: bare-stencil coastline
    injection, most acute near the equator where no Coriolis cages it).

    PREVIOUS FORM (replaced — the day-75+ Amazon-fan eta blob): F = -grad(
    p_bc_avg)/RHO_0 with p_bc_avg the H_sw-normalized trapezoidal average of
    p_bc over ALL 13 layers including ghost water. Because p_bc is CONSTANT
    below the seafloor (rho is masked before the cumsum, so dp=0 there), the
    ghost part contributes ((H_sw-H)/H_sw) * grad(p_bc_bottom) to the
    forcing — at shelf breaks grad(p_bc_bottom) is O(5e3 Pa / 1e5 m), so the
    ghost term alone is O(3e-5 m/s2). It displaces the implied equilibrium
    sea level eta_eq = -p_bc_avg/(g*rho0) by O(1-2 m) between adjacent cells
    across every isobath step, and the free-surface step piles eta up at each
    jump; near the equator (f->0) Coriolis cannot geostrophically cage the
    pile-up -> eta ran 2.5 -> 12.25 m in 15 d at (47.5W, 7.5N) on the 2000 m
    isobath off the Amazon fan. The transport-weighted form contains no ghost
    water and no below-bottom constant, so the spurious isobath forcing
    vanishes; the remaining wet-column form stress is the physical (JEBAR-type)
    coupling, ~10x smaller at the blob site.

    Measured at the d75 blob state, cell (311,66), H=2000 m, H_sw=4000 m:
    |F_old| = 4.6e-5 -> |F_new| = 4.9e-6 m/s2; global mean |F| 1.3e-5 ->
    6.2e-6 m/s2.
    """
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
    """FD analogue of the spectral baseline's _dealias_h.

    Nonlinear products in physical space (u*du/dx, T*div_h, ...) and the
    central-difference divergence that diagnoses w all amplify the 2-dx
    grid-scale mode. The spectral solver removes it with a 2/3 FFT rule on
    BOTH horizontal axes (jax_solver.py:450). Here the lon axis is periodic
    + uniformly spaced, so the 2/3 FFT rule is exact and faithful. The lat
    axis is NOT periodic (closed no-flux N/S walls) and NOT uniformly spaced
    on a lat-lon grid, so an FFT there would mangle the field — instead a
    5-pt binomial low-pass [1,4,6,4,1]/16 substitutes: its transfer function
    is 0 at the Nyquist (2-dx) wavenumber, ~0.01 near-Nyquist, and ~0.92 at
    8-dx, so it kills the grid scale while preserving resolvable structure
    (verified). edge-padding mirrors the wall ghost cell, consistent with
    _d_dy/_laplacian_h. This is applied to w AND to every nonlinear
    advection tendency (adv_u, adv_v, adv_T), matching the spectral
    baseline's call sites (jax_solver.py:490, 515, 757).
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

    w = -Fz (the exact interface transports of _vertical_transport_iface,
    sign-flipped to the w>0 = DOWNWARD convention): the trapezoid node-w
    built from div_avg + cumsum + dealias could not invert the advection's
    discrete face-flux divergence pointwise (residual up to 1.2e-5/s; a
    T-proportional source, i.e. a 4%/step exponential pump — max|T| 29.6 ->
    52.6 in 30 steps at the North Brazil Current node). The exact inverse
    closes div_x + div_y + div_z to machine precision for a uniform tracer.
    No dealiasing: dealiasing w would re-open the discrete non-closure the
    exact inverse just fixed.
    """
    Fz = _vertical_transport_iface(state.u, state.v, p)
    w = -Fz[..., :-1] * p.wet_mask_z
    return w


# ── Tendencies (FD, with land masking) ─────────────────────────────

def _gradient_face_gated_3d(field, p):
    """Face-gated horizontal gradient for advection: d/dx, d/dy with each
    face difference zeroed at every wet/ghost (or wet/dry) interface.

    The bare centered _d_dx/_d_dy differences across mask boundaries: the
    ghost nodes hold the T_ref sentinel (+15 C, set in init_state) while
    real 4000 m water is ~+1 C, so the centered stencil reads dT ~ 14 K
    across EVERY wet/ghost face. A 6 cm/s deep coastal current then feels
    v*dT/dy ~ 0.3 K/d of advection from a temperature that does not exist —
    a linear pump adv ∝ (15 - T_wet) that drove the k12/k13 warm/cold dipole
    (130,19) from d50 to the d360 max|T| = 1247 C blow-up (adv-form runs
    gpu365_cap3d, ctl290nogm: FAIL_DRIFT both, only adv pumps; terms_fn
    showed adv -0.24 K/d at k13 growing linearly with the anomaly, all
    other terms < 0.03). Zeroing the face difference at mask boundaries is
    consistent with the ALREADY face-gated mass flux (the divergence in
    _divergence_conservative zeroes tracer flux across closed faces) and
    with _gradient_conservative_3d (used for the PGF for the same reason).
    Where the flow is parallel to the mask boundary (u|face=0, v|face=0),
    the cross-boundary advective flux is zero anyway — the gated gradient
    removes only the unphysical part.

    Returns (dF/dx, dF/dy) on wet nodes; ghost nodes keep whatever the
    caller masks away.
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

    Advective form (not flux form) avoids the spurious u*div_h source.
    Horizontal gradients use _gradient_face_gated_3d (zero face difference
    at wet/ghost mask boundaries — the ghost sentinel cliff otherwise acts
    as a spurious advective pump; see there for the d360 blow-up trail).
    Vertical gradient stays the bare _d_dz: w is already masked to zero in
    ghost layers and the tracer terms multiply by wet_mask_z, so a wet/ghost
    vertical face carries no advective flux regardless of the dT/dz it reads.
    Nonlinear products (u*du/dx etc.) are
    2/3-rule-dealiased via _dealias_h_fd, matching the spectral baseline
    (jax_solver.py:490): the summed tendency is dealiased once, then
    land-masked. Without this the aliasing of the nonlinear products
    injects 2-dx energy that drives the boundary heat pump (see
    _dealias_h_fd).
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


def _advection_scalar(T, u, v, Fz_in, p):
    """3D FLUX-FORM scalar advection (FD, land-masked). Linear -> no dealias.

    Switched from advective form (−u·∇T) to flux form (−∇·(uT)) after the
    real-grid dt=3600 split run: the advective form equals flux form minus
    T·∇·u, and the +T·∂u/∂x part is a REAL anti-diffusion — at any discrete
    divergence (equatorial upwelling, ∂w/∂z>0 in the 5 m layer) the operator
    carries a genuine positive eigenvalue dt·δ per step that grows the field
    exponentially. Subcycling (adv_nsub) slows the per-step growth but the
    eigenvalue scales with dt·δ and survives any n: the (310.5E, 6.5N)
    surface T runaway (+3.4e-3 K/s growing 20x in 2 steps, NaN by step 23)
    persisted through adv_nsub=6. Flux form has no anti-diffusive term — the
    centered-flux operator is neutrally transportive — and is exactly
    conservative, matching the continuity-consistent tracer equation the
    free-surface subcycle already solves. The advective form remains in
    _advection_flux_form for MOMENTUM (vector-invariant scheme needs it).
    Horizontal fluxes are face-gated (see _gradient_face_gated_3d): the ghost
    sentinel T=15 C at mask boundaries otherwise reads as a 14 K step-gradient
    advected into the wet interior. The vertical flux uses the interface
    transports Fz_in (nx, ny, nz+1) from _vertical_transport_iface — the
    EXACT discrete inverse of the horizontal divergence — with donor-cell
    (upwind) T face values: the bottom face reads _fill_ghost_bottom(T) so
    the bottom-wet layer reads the no-flux bottom BC; Fz_in is 0 below the
    seafloor (masked u,v) anyway.
    """
    wm = p.wet_mask_z
    # ── Zonal flux at face (i+1/2), periodic in x ──
    # Fx = u_face * T_face, face-gated (both cells wet). Centered face values.
    # Sign: Fx is the +x-directed flux (u>0 carries T eastward).
    Tx_face = 0.5 * (T + jnp.roll(T, -1, axis=0))
    ux_face = 0.5 * (u + jnp.roll(u, -1, axis=0))
    gate_x = wm * jnp.roll(wm, -1, axis=0)
    Fx = ux_face * Tx_face * gate_x                    # (nx, ny, nz)
    # ── Meridional flux at face (j+1/2), closed N/S walls ──
    # Spherical conservation: the face MASS flux carries cos(face_lat), and
    # the cell divergence divides by cos(cell lat) (see
    # _divergence_conservative for the telescoping argument). T flux = v*T
    # gets the same cos_face factor.
    pad = [(0, 0), (1, 1), (0, 0)]
    T_pad = jnp.pad(T, pad, mode='edge')
    v_pad = jnp.pad(v, pad, mode='edge')
    wm_pad = jnp.pad(wm, pad, mode='edge')
    Ty_face = 0.5 * (T_pad[:, 1:-1] + T_pad[:, 2:])
    vy_face = 0.5 * (v_pad[:, 1:-1] + v_pad[:, 2:])
    gate_y = wm_pad[:, 1:-1] * wm_pad[:, 2:]
    cos_face = 0.5 * (p.cos_lat + jnp.roll(p.cos_lat, -1))
    Fy = vy_face * Ty_face * gate_y * cos_face[None, :, None]   # (nx, ny, nz)
    # Close the truncation-wall faces: the edge pad replicates wm at j=ny-1,
    # so gate_y there is wm^2 = 1 and Fy[:, -1] would carry a flux through the
    # closed wall (v is masked to 0 at boundary rows, but vy_face reads the
    # INTERIOR neighbor 0.5*v[ny-2]). Explicit closure, matching
    # _divergence_conservative's vface_open/fcos_in wall sets.
    Fy = Fy.at[:, -1].set(0.0)
    # ── Vertical flux at interface (k+1/2): DONOR-CELL (upwind) ──
    # Sign convention (from _vertical_transport_iface): Fz is the
    # DOWNWARD-positive interface volume transport (nx, ny, nz+1),
    # Fz[..., nz] = 0 at the seafloor, built by the EXACT discrete inverse
    # of the horizontal divergence (div_x + div_y + div_z = 0 to machine
    # precision for a uniform tracer). The donor for an interface flux is
    # the SHALLOWER node when Fz>0 (water arrives from above) and the
    # DEEPER node when Fz<0 (physical upwelling carries the deep value up).
    # Two reasons the T face value must not be centered:
    #   1. RK2/forward-Euler time integration of a CENTERED vertical
    #      transport is unconditionally unstable (|lam|^2 = 1+theta^2 > 1 for
    #      any theta) — at dt=3600 the 5 m layer's theta_v = dt*w/dz = 1.3-1.8
    #      grew ~2x/step (the step-21 equatorial T runaway that survived
    #      adv_nsub=6 as a still-growing 1.08x/step). Donor-cell adds a
    #      monotone scheme: stable for theta <= 1 per substep, with added
    #      vertical diffusivity 0.5*|w|*dz <= 6e-3 m^2/s (below kappa_v=1e-5).
    #   2. The centered T face value multiplies the FULL tracer content into
    #      the surface node's flux (w*T/dz ~ O(10 K/step) wherever the
    #      column-divergence leak gives w[0] != 0) — the old advective form
    #      was protected by the small dT/dz factor; donor-cell keeps the
    #      flux one-sided (physical upwelling carries the DEEP value up,
    #      surface convergence carries the SURFACE value down).
    # Bottom face: _fill_ghost_bottom(T) makes the bottom-wet layer read the
    # no-flux seafloor BC; Fz is 0 below the seafloor (masked u,v) anyway.
    T_deep = _fill_ghost_bottom(T, p)[..., 1:]        # node k (below iface k)
    T_shallow = T[..., :-1]                           # node k-1 (above iface k)
    wet_iface = wm[..., :-1] * wm[..., 1:]
    # Internal interfaces are Fz_in[..., 1:-1] (indices 1..nz-1, count nz-1):
    # index 0 is the surface (rigid-lid leak, handled below) and index nz is
    # the seafloor (zero transport).
    Fz_int = Fz_in[..., 1:-1] * jnp.where(Fz_in[..., 1:-1] > 0.0, T_shallow, T_deep) * wet_iface
    # Padded flux array: Fz_top = 0 (rigid lid, w=0), Fz_bot = 0 (seafloor).
    # tend[k] = -(dF/dx + dF/dy + dF/dz) with the vertical term
    # (Fz[k+1/2] - Fz[k-1/2]) / dz_node[k]: k increases DOWNWARD, so the
    # out-face of cell k is its BOTTOM face (dn, at k+1/2) and the in-face is
    # the top (up, at k-1/2) — the same out-minus-in form as div_x
    # (Fx[i+1/2] - Fx[i-1/2]). Writing this (up - dn) instead silently makes
    # the donor-cell scheme ANTI-upwind: T_new = (1+theta)*T - theta*T_donor,
    # a +theta eigenvalue per substep (measured: T_w grew 29.6 -> 2350 over
    # 6 substeps, factor 2.45 = 1+theta_v with theta_v=1.453 in the 5 m
    # layer). Fz[...] is a DOWNWARD-positive flux (w>0 = down, donor above).
    Fx_up = jnp.roll(Fx, 1, axis=0)                   # flux at face (i-1/2)
    Fy_up = jnp.roll(Fy, 1, axis=1)
    Fy_up = Fy_up.at[:, 0].set(0.0)                   # north wall closed
    # Top face: Fz[..., 0] is the column-integrated horizontal divergence —
    # the rigid-lid leak the barotropic subcycle simultaneously absorbs as
    # the eta tendency. Left out of the budget it
    # is a T-INDEPENDENT inflow of T_deep into the surface node wherever
    # Fz[0] != 0 (measured: +16.47 K per 600 s substep, linear forever). The
    # MOM-style rigid-lid closure is a top-face flux carrying the surface
    # cell's own value: Fz_top = Fz[0]*T[0]. Under upwelling (Fz[0]<0) the
    # surface node then equilibrates at the deep value (physical upwelling
    # replacement) instead of growing without bound; under surface
    # convergence (Fz[0]>0) the donor from "above" is the edge-extended
    # surface value, so the node sees pure through-flow (no change).
    Fz_top = Fz_in[:, :, :1] * T[..., :1]
    up = jnp.concatenate([Fz_top, Fz_int], axis=-1)
    dn = jnp.concatenate([Fz_int, jnp.zeros_like(Fz_int[..., :1])], axis=-1)
    div_x = (Fx - Fx_up) * p.inv_dx[..., 0:1]
    # Meridional: divide by cos(cell lat) like _divergence_conservative.
    div_y = (Fy - Fy_up) * p.inv_dy / p.cos_lat[None, :, None]
    div_z = (dn - up) / p.dz_node
    adv_T = -(div_x + div_y + div_z)
    return adv_T * p.wet_mask_z


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
    diff_v_u = p.nu_v * _d2_dz2(state.u, p)
    diff_v_v = p.nu_v * _d2_dz2(state.v, p)

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
# Represents unresolved baroclinic eddies as an advective bolus transport
# that flattens isopycnal slopes, releasing baroclinic available potential
# energy (APE). Required at coarse (1°) resolution where the baroclinic
# Rossby radius (~30-50km) is sub-grid; the explicit RK2 residual PGF
# otherwise pumps energy into unresolvable internal-gravity-wave modes.
# Tracer-only (not applied to momentum); gated by p.kappa_gm > 0.

# Vertical stratification floor [kg/m^4] guards ∂rho'/∂z against division
# blow-up in weakly-stratified / convective columns ( rho' ~ well-mixed ).
_GM_RHOZ_FLOOR = 1.0e-5

# Target value of dt*D_v*(1/h_k + 1/h_k1)/dz_iface (explicit-vertical-diffusion
# CFL number) for the Redi S^2 vertical skew term, per interface. See the
# comment at the cap site in _redi_skew_flux_tendency.
_REDI_CFL_TARGET = 0.4


def _isopycnal_slope(state, p):
    """Isopycnal slope S = (S_x, S_y) = -∇_h(rho') / ∂rho'/∂z.

    With z increasing downward (index 0 = surface), a stable column has
    rho' increasing with depth, so ∂rho'/∂z > 0 and the slope points
    down the horizontal density gradient (toward denser water).

    The raw slope is tanh-clipped to ±gm_slope_max to avoid singularity
    where stratification is weak; the stratification denominator is
    floored at _GM_RHOZ_FLOOR before the clip. Returns (S_x, S_y),
    each (nx, ny, nz), masked to wet points.
    """
    rho_prime = _density_anomaly(state.T, state.S, p) * p.wet_mask_z
    # Seafloor no-flux fill for the VERTICAL density gradient: the raw
    # masked field is 0 in ghost layers, so the bottom wet layer's _d_dz
    # sees (0 - rho_bottom) < 0 — an INVERTED column that floored-but-
    # sign-preserved makes the closure read "convective" at every sill
    # (~11.7k columns) and pin |S| at the clip there. Filling the ghost
    # with the bottom wet value makes drho_dz at kbot read ~0 (one-sided,
    # zero gradient across the floor) — the physical no-flux seafloor BC.
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
    # Danabasoglu-McWilliams (1995) slope taper: sigma = 1/(1+(|S|/S_lim)^4).
    # The closure flux is multiplied by sigma, which -> 1 for |S| << S_lim
    # (stratified interior: full GM) and -> 0 for |S| >> S_lim (weakly
    # stratified deep ocean / steep fronts: closure suppressed).
    # A tanh CLIP is the wrong treatment: it SATURATES the slope at S_lim,
    # holding the flux at full kappa*S_lim^2 (an effective 0.1 m^2/s vertical
    # diffusivity at kappa_gm=1000) throughout the weakly-stratified deep
    # ocean — a spurious diapycnal pump that erodes deep stratification,
    # drives the Southern-Ocean deep-T runaway, and drains the subtropical
    # gyres (-6 m/yr eta trend, linear in kappa_gm; 365d FAIL at all
    # kappa_gm in {300,1000}, with/without kappa_redi). The DM95 taper is
    # the standard OGCM treatment (also Large et al. 1997).
    s_max = p.gm_slope_max
    S2 = S_x * S_x + S_y * S_y
    s4 = (s_max * s_max) ** 2
    sigma = 1.0 / (1.0 + (S2 * S2) / s4)
    S_x = sigma * S_x
    S_y = sigma * S_y
    return S_x * p.wet_mask_z, S_y * p.wet_mask_z


def _gm_bolus_velocity(state, p):
    """Eddy-induced (bolus) transport velocity (u*, v*, w*).

    u*, v* = -κ_GM · S  (down the isopycnal slope, flattening density).
    w* is diagnosed from horizontal continuity of (u*, v*) — same pattern
    as _compute_vertical_velocity — so the bolus is non-divergent in the
    interior (mass-conservative tracer transport). All three are masked
    to wet points; zero when the closure is off.
    """
    if p.kappa_gm <= 0.0:
        z = jnp.zeros_like(state.u)
        return z, z, z
    S_x, S_y = _isopycnal_slope(state, p)
    u_star = -p.kappa_gm * S_x
    v_star = -p.kappa_gm * S_y
    # Vertical bolus from continuity: ∂w*/∂z = -(∂u*/∂x + ∂v*/∂y), w*=0 at bottom.
    div_star = _divergence_h(u_star, v_star, p)
    div_avg = 0.5 * (div_star[..., :-1] + div_star[..., 1:])
    integrand = div_avg * p.dz_3d
    w_star = jnp.zeros_like(state.u)
    w_star = w_star.at[..., :-1].set(
        -jnp.cumsum(integrand[..., ::-1], axis=-1)[..., ::-1])
    w_star = _dealias_h_fd(w_star, p)
    return (u_star * p.wet_mask_z, v_star * p.wet_mask_z,
            w_star * p.wet_mask_z)


def _gm_tracer_transport(tracer, u_star, v_star, w_star, p):
    """Bolus advection tendency -(u*·∂T/∂x + v*·∂T/∂y + w*·∂T/∂z).

    Structurally identical to _advection_scalar but driven by the bolus
    velocity: face-gated horizontal gradients (the ghost sentinel cliff is
    unphysical for the bolus too — u* is nonzero at mask boundaries and
    dT ~ 14 K across a wet/ghost face would pump the same way), bare _d_dz
    vertically. Dealiased and land-masked. Returns (nx, ny, nz) tendency.
    """
    dT_dx, dT_dy = _gradient_face_gated_3d(tracer, p)
    dT_dz = _d_dz(_fill_ghost_bottom(tracer, p), p)
    gm_T = -(u_star * dT_dx + v_star * dT_dy + w_star * dT_dz)
    gm_T = _dealias_h_fd(gm_T, p)
    return gm_T * p.wet_mask_z


# ── Redi isopycnal mixing (dissipative counterpart to GM) ──────────
# Full isoneutral diffusion flux (z-up convention, matching _d_dz which
# returns dC/dz_up): the along-isopycnal gradient is
#     ∇_ρ C = ∇_h C + S ∂_z C        (S = -∇_h ρ / ∂_z ρ, points down-slope)
# and the Redi flux F = -κ_iso ∇_ρ C decomposes as
#     F^h = -κ (∇_h C + S ∂_z C)           (horizontal component)
#     F^z = -κ (S·∇_h C + |S|² ∂_z C)      (vertical component)
# The -κ ∇_h C piece equals the background κ_h ∇²_h C already applied by the
# linear step (and subtracted back in _compute_tracer_residual), so in Griffies
# skew-flux RESIDUAL form we drop it and keep the slope-driven remainder:
#     F^h_skew = -κ S ∂_z C
#     F^z_skew = -κ (S·∇_h C + |S|² ∂_z C)
# The vertical S·∇_h C cross-term is essential: without it the horizontal
# skew flux alone can STEEPEN a front (it advects down the slope). The full
# residual is guaranteed dissipative (it is a true along-isopycnal diffusion
# minus the redundant horizontal part). Tendency = -∇·F^skew.
# Dealiased + land-masked.

def _redi_skew_flux_tendency(tracer, S_x, S_y, p, kappa=None):
    """Isopycnal skew-flux residual tendency for one tracer.

    This is the STANDARD GM/Redi closure form (Griffies 1998): the
    bolus-transport + isoneutral-diffusion pair is recast as a single
    skew-flux tensor, whose vertical term is DIFFUSIVE (CFL ~ kappa*|S|^2*
    dt/dz^2) rather than advective (CFL ~ |w*|*dt/dz). On a non-uniform
    vertical grid with a thin surface layer (dz=5 m), the advective bolus
    form has CFL = 0.29*60/5 = 3.49 (unstable); the skew-flux form has
    CFL = 1000*0.01^2*60/5^2 = 0.48 (stable). Same closure, different
    discretization — the skew-flux form is what MOM6/MITgcm/NEMO use.

    Args:
        tracer: (nx, ny, nz) T or S.
        S_x, S_y: (nx, ny, nz) isopycnal slopes from _isopycnal_slope.
        kappa: diffusivity [m^2/s]; if None, reads p.kappa_redi. Pass
            p.kappa_gm explicitly to use this same operator for the GM
            closure (the mathematically equivalent skew-flux form).
    Returns: (nx, ny, nz) tendency dC/dt, dealiased and land-masked.
    Zero when the effective kappa <= 0.
    """
    if kappa is None:
        kappa = p.kappa_redi
    if kappa <= 0.0:
        return jnp.zeros_like(tracer)
    k = kappa
    # Seafloor no-flux fill for the vertical derivatives: _d_dz's centered
    # stencil at the bottom wet layer reaches into the ghost layer, which
    # holds T_ref = 15 C forever — a spurious ~0.08 K/day seafloor heat
    # flux at sill columns that fed the Southern-Ocean deep runaway. Fill
    # ghosts with the bottom wet value (no-flux BC) before differentiating.
    tracer = _fill_ghost_bottom(tracer, p)
    dC_dx, dC_dy = _gradient_conservative_3d(tracer * p.wet_mask_z, p)
    dC_dz = _d_dz(tracer, p)
    # Horizontal skew flux F_h = -k * S * dC/dz  (x,y components)
    Fx = -k * S_x * dC_dz
    Fy = -k * S_y * dC_dz
    tend_h = -_divergence_conservative_3d(Fx, Fy, p)
    # Vertical skew flux, INTERFACE flux form (MOM6/NEMO discretization):
    # the flux lives on interfaces k+1/2 (between nodes k and k+1); each
    # interface uses the ONE-SIDED cell values and the interface slope
    # S(k+1/2) = 0.5*(S[k] + S[k+1]). The tendency of cell k is
    # (F[k-1/2] - F[k+1/2]) / dz[k] — a compact 3-point stencil, exactly
    # diffusive (negative-semidefinite for the S2 term), with the flux
    # zero at the top/bottom material boundaries.
    # The previous NODE-flux form (Fz at nodes from centered dC/dz, then
    # _d_dz of Fz) is a 5-point/2-step stencil on which the even/odd
    # sublattices DECOUPLE: the sawtooth-in-z mode has eigenvalue exactly 0
    # (never damped) and the operator is asymmetric (max asym 2.7e-2) with
    # mildly positive symmetric-part eigenvalues on the stretched deep grid
    # — it pumped a deep-T runaway linear in kappa_gm and gm_slope_max
    # (365d FAIL at d270-310 for tanh-clip / DM95-taper / slope-max 0.001).
    Cm = tracer[..., :-1]                        # upper cell value
    Cp = tracer[..., 1:]                         # lower cell value
    dC_dz_iface = (Cp - Cm) / p.dz_iface         # (nx, ny, nz-1)
    S_x_i = 0.5 * (S_x[..., :-1] + S_x[..., 1:])
    S_y_i = 0.5 * (S_y[..., :-1] + S_y[..., 1:])
    dC_dx_i = 0.5 * (dC_dx[..., :-1] + dC_dx[..., 1:])
    dC_dy_i = 0.5 * (dC_dy[..., :-1] + dC_dy[..., 1:])
    S2_i = S_x_i * S_x_i + S_y_i * S_y_i
    # Explicit-CFL cap on the vertical skew diffusivity. The S^2 term is an
    # explicit interface diffusion with D_v = k*|S|^2; its two-cell
    # stability bound (Gershgorin on the k/k+1 pair) is
    #   dt * D_v * (1/dz_k + 1/dz_k1) / dz_iface  <=  ~2  (RK2)
    # On a stretched grid with a 5 m surface layer, k=1000 and slopes at
    # the DM95 taper cap, dt*D_v/dz^2 reaches 14 at the default
    # gm_slope_max=0.01 and dt=3600 s — the Gulf Stream front blew up at
    # step ~857 of the mode-split run (stable for 1000+ steps at slope cap
    # 0.001, i.e. exactly when this CFL drops below ~0.9). The stratified
    # interior (D_v << bound) is untouched; only thin-layer steep-front
    # corners are clipped. This is the MOM6-style dt-dependent limiting of
    # the vertical Redi diffusivity.
    dzu = p.dz_node[..., :-1]                    # upper cell thickness h_k
    dzl = p.dz_node[..., 1:]                     # lower cell thickness h_k1
    D_v_max = _REDI_CFL_TARGET * p.dz_iface / (p.dt * (1.0 / dzu + 1.0 / dzl))
    S2_eff = jnp.minimum(S2_i, D_v_max / k)
    Fz_i = -k * (S_x_i * dC_dx_i + S_y_i * dC_dy_i + S2_eff * dC_dz_iface)
    # Material boundaries: no isopycnal transport crosses the surface, the
    # seafloor, or a land/rock wall — zero the flux on any interface where
    # either adjacent node is dry (wet_iface covers seafloor + coastal sills;
    # the ghost fill above already removed the T_ref=15 reservoir from the
    # one-sided differences, so flux into ghost columns is harmless: those
    # cells are re-masked to 0 by the final wet_mask_z multiply anyway).
    wet_iface_f = (p.wet_mask_z[..., :-1] > 0.5) & (p.wet_mask_z[..., 1:] > 0.5)
    Fz_i = jnp.where(wet_iface_f, Fz_i, 0.0)
    # NOTE: the material top/bottom boundaries are NOT Fz_i[0]/Fz_i[-1] —
    # those are INTERIOR interfaces (node 0↔1, node 12↔13). Zeroing them
    # decouples the surface/bottom nodes from vertical skew transport
    # entirely (found by eigenvector test: null vector was e_0). The
    # padded flux array below already carries the zero-flux BC.
    # Tendency of cell k: (F[k-1/2] - F[k+1/2]) / dz[k], z-up convention
    # (positive Fz_i = upward transport). Interface array Fz_i has indices
    # 0..nz-2 = interfaces (1/2 .. nz-3/2); cell k sits between interfaces
    # k and k+1 of the padded array F[0..nz] with F[0]=F[nz]=0:
    #   F[k]   = Fz_i[k-1]  (k>=1)        F[k+1] = Fz_i[k]  (k<=nz-2)
    # so tend_v[k] = (Fz_i[k-1] - Fz_i[k]) / dz[k] with edge zeros.
    up = jnp.concatenate([jnp.zeros_like(Fz_i[..., :1]), Fz_i], axis=-1)   # Fz_i[k-1]
    dn = jnp.concatenate([Fz_i, jnp.zeros_like(Fz_i[..., :1])], axis=-1)   # Fz_i[k]
    tend_v = (up - dn) / p.dz_node
    tend = tend_h + tend_v
    tend = _dealias_h_fd(tend, p)
    return tend * p.wet_mask_z


def _tracer_terms(state, p):
    """Diagnostic decomposition of dT/dt into physical terms.

    Mirrors _compute_tracer_tendency exactly (same expressions), returning
    each term separately: [adv, diff_h, diff_v, conv, gm, redi]. For offline
    blowup attribution (which closure pumps the deep-T runaway). NOT used in
    the time integration — read-only diagnosis.
    """
    Fz = _vertical_transport_iface(state.u, state.v, p)
    adv_T = _advection_scalar(state.T, state.u, state.v, Fz, p)
    diff_h_T = p.kappa_h * _laplacian_h(state.T, p)
    diff_v_T = _diff_v_flux_tendency(state.T, p.kappa_v, p)

    rho_prime = _density_anomaly(state.T, state.S, p) * p.wet_mask_z
    wet_iface = (p.wet_mask_z[..., :-1] > 0.5) & (p.wet_mask_z[..., 1:] > 0.5)
    unstable_iface = (rho_prime[..., :-1] > rho_prime[..., 1:]) & wet_iface
    conv_mask_3d = jnp.any(unstable_iface, axis=-1, keepdims=True)
    conv_T = _conv_flux_tendency(state.T, conv_mask_3d, p.kappa_conv, p)

    if p.kappa_gm > 0.0:
        S_x_gm, S_y_gm = _isopycnal_slope(state, p)
        gm_T = _redi_skew_flux_tendency(state.T, S_x_gm, S_y_gm, p,
                                        kappa=p.kappa_gm)
    else:
        gm_T = jnp.zeros_like(state.T)
    if p.kappa_redi > 0.0:
        S_x, S_y = _isopycnal_slope(state, p)
        redi_T = _redi_skew_flux_tendency(state.T, S_x, S_y, p)
    else:
        redi_T = jnp.zeros_like(state.T)

    terms = [adv_T, diff_h_T, diff_v_T, conv_T, gm_T, redi_T]
    return jnp.stack([t * p.wet_mask_z for t in terms], axis=0)


def _compute_tracer_tendency(state, p):
    """dT/dt, dS/dt (FD, land-masked). Includes bulk air-sea heat flux.

    Vertical advection is SUBCYCLED adv_nsub times (frozen velocity, frozen
    horizontal terms) when adv_nsub > 1: the explicit advection CFL is
    dt*(|u|/dx + |v|/dy + w/dz) < ~1, and at dt=3600 s the 5 m surface layer
    with equatorial upwelling w~2e-3 m/s gives dt*w/dz = 1.3-1.8 > 1 (the
    real-grid split run ran away +0.3 K/step at (310.5E, 6.5N) from step 21
    and NaN'd by step 29). At the historical dt=60-300 the CFL was 0.02-0.18
    and a single application was fine (adv_nsub=1 keeps that path untouched).
    Only the ADVECTIVE term is subcycled: diffusion (kappa_h dt/dz² << 0.5),
    surface heat/salt fluxes (dt-weighted, non-CFL), GM/Redi skew flux
    (CFL 0.58 at dt=3600), and convective adjustment (own conv_nsub) are all
    inside their explicit bounds at dt=3600 and stay evaluated ONCE —
    subcycling them too would cost wall-clock for nothing. The velocity and
    the horizontal structure are frozen over the substeps (fixed-operator
    subcycling, same pattern as conv_nsub); only the advected FIELD evolves,
    which is what the CFL constrains.
    """
    Fz = _vertical_transport_iface(state.u, state.v, p)
    n_a = int(p.adv_nsub)
    if n_a > 1 and p.use_scan:
        # Fused adv subcycle (numerically identical to the unrolled loop).
        dts = p.dt / n_a

        def _adv_sub(carry, _):
            tT, tS = carry
            aT = _advection_scalar(tT, state.u, state.v, Fz, p)
            aS = _advection_scalar(tS, state.u, state.v, Fz, p)
            return (tT + aT * dts, tS + aS * dts), (aT, aS)

        (T_w, S_w), (sT, sS) = jax.lax.scan(
            _adv_sub, (state.T, state.S), None, length=n_a)
        adv_T, adv_S = sT.mean(axis=0), sS.mean(axis=0)
    elif n_a > 1:
        # Subcycled advection: evolve a working copy of (T, S) with the
        # frozen u,v,Fz advection operator dt/adv_nsub at a time, and report
        # the AVERAGE rate so the Strang residual (which subtracts it from
        # the full tendency and re-applies it over the full dt in the N-step
        # RK2) integrates the same subcycled operator over dt.
        dts = p.dt / n_a
        T_w, S_w = state.T, state.S
        adv_T_sum = jnp.zeros_like(T_w)
        adv_S_sum = jnp.zeros_like(S_w)
        for _ in range(n_a):
            aT = _advection_scalar(T_w, state.u, state.v, Fz, p)
            aS = _advection_scalar(S_w, state.u, state.v, Fz, p)
            adv_T_sum = adv_T_sum + aT
            adv_S_sum = adv_S_sum + aS
            T_w = T_w + aT * dts
            S_w = S_w + aS * dts
        adv_T = adv_T_sum / n_a
        adv_S = adv_S_sum / n_a
    else:
        adv_T = _advection_scalar(state.T, state.u, state.v, Fz, p)
        adv_S = _advection_scalar(state.S, state.u, state.v, Fz, p)

    diff_h_T = p.kappa_h * _laplacian_h(state.T, p)
    diff_h_S = p.kappa_h * _laplacian_h(state.S, p)
    diff_v_T = _diff_v_flux_tendency(state.T, p.kappa_v, p)
    diff_v_S = _diff_v_flux_tendency(state.S, p.kappa_v, p)

    # Convective adjustment (same logic as spectral; inert under linear EOS
    # + stable heating, but kept for consistency).
    # CRITICAL: mask ghost water AND require BOTH adjacent layers wet before
    # flagging an interface unstable. The bottommost wet layer typically has
    # rho'>0 (cold/salty deep water) while the ghost layer beneath is masked
    # to rho'=0; without the both-wet guard, every wet/ghost interface tests
    # as "unstable" (rho_bottom > 0 = rho_ghost), convects the WHOLE column
    # (the any(axis=-1) mask), and seeds a ~3e-3 K/s surface-T blowup (282
    # K/day) -> NaN by day 3. 81% of "unstable" interfaces were this ghost
    # artifact (14931/18423). Masking rho' alone is not enough: a wet layer
    # over a ghost layer (rho 0) still compares to 0.
    rho_prime = _density_anomaly(state.T, state.S, p) * p.wet_mask_z
    wet_iface = (p.wet_mask_z[..., :-1] > 0.5) & (p.wet_mask_z[..., 1:] > 0.5)
    unstable_iface = (rho_prime[..., :-1] > rho_prime[..., 1:]) & wet_iface
    conv_mask_3d = jnp.any(unstable_iface, axis=-1, keepdims=True)
    # kappa_conv CFL: kappa_conv*dt/dz_top² at dt=3600 s on the 10 m surface
    # layer = 1.8 >> 0.5 (explicit diffusion limit). Subcycling the LINEAR
    # convective operator conv_nsub times with kappa_conv/conv_nsub each is
    # mathematically identical to one kappa_conv application (fixed mask,
    # linear operator) but restores the explicit stability bound per substep.
    # The mask stays frozen over the baroclinic step — the adjustment still
    # removes the same instability energy; only the trajectory differs O(dt).
    n_c = int(p.conv_nsub)
    if n_c > 1:
        kappa_c = p.kappa_conv / n_c
        h_c = p.dt / n_c
        conv_T = jnp.zeros_like(state.T)
        conv_S = jnp.zeros_like(state.S)
        T_c, S_c = state.T, state.S
        if p.use_scan:
            # Fused conv subcycle (numerically identical forward-Euler chain).
            def _conv_sub(carry, _):
                tT, tS = carry
                ttT = _conv_flux_tendency(tT, conv_mask_3d, kappa_c, p)
                ttS = _conv_flux_tendency(tS, conv_mask_3d, kappa_c, p)
                return (tT + ttT * h_c, tS + ttS * h_c), (ttT, ttS)

            (T_c, S_c), (sT, sS) = jax.lax.scan(
                _conv_sub, (state.T, state.S), None, length=n_c)
            conv_T, conv_S = sT.mean(axis=0), sS.mean(axis=0)
        else:
            for _ in range(n_c):
                # Advance by the CURRENT term only (correct forward-Euler
                # subcycle, same pattern as the adv_nsub loop above). Advancing
                # by the accumulated sum makes the operator NEUTRAL per
                # eigenmode (|1 - hd/2 ± i.sqrt(hd)| = 1 for hd < 4) instead of
                # dissipative: inversions never homogenize, and the reported
                # average rate (sum of accumulated terms)/n_c takes a
                # phase-random sign that pumps ~0.2x the mode amplitude per
                # step (measured conv-max growth 9.5e-4 -> 5.4e-3 -> 0.14 ->
                # 2.6 K/s over 3 steps at the warm pool, ~5x/step).
                tT = _conv_flux_tendency(T_c, conv_mask_3d, kappa_c, p)
                tS = _conv_flux_tendency(S_c, conv_mask_3d, kappa_c, p)
                conv_T = conv_T + tT
                conv_S = conv_S + tS
                T_c = T_c + tT * h_c
                S_c = S_c + tS * h_c
            conv_T = conv_T / n_c   # average rate over the baroclinic step
            conv_S = conv_S / n_c
    else:
        conv_T = _conv_flux_tendency(state.T, conv_mask_3d, p.kappa_conv, p)
        conv_S = _conv_flux_tendency(state.S, conv_mask_3d, p.kappa_conv, p)

    heat_factor = 1.0 / (RHO_0 * C_P * p.dz_surface)
    heat_T = p.Q_heat_2d[:, :, None] * heat_factor * p.surface_mask

    # Bulk air-sea heat flux (Haney/Barnier): genuine SST negative feedback.
    bulk_T = (p.lambda_bulk * (p.T_atm_3d - state.T[:, :, 0:1])
              * heat_factor * p.surface_mask)

    # Surface salinity restoring (Haney): equivalent salt flux relaxing SSS
    # to climatology with timescale tau = 1/restore_coef_S. Same form as the
    # bulk heat flux — a surface tracer BC, not a body source (surface_mask).
    rest_S = (p.restore_coef_S * (p.S_ref_2d[:, :, None] - state.S[:, :, 0:1])
              * p.surface_mask)

    # Gent-McWilliams sub-grid baroclinic closure. Recast in skew-flux
    # residual form (Griffies 1998): the bolus-transport + isoneutral-
    # diffusion pair is a single skew-flux tensor whose vertical term is
    # DIFFUSIVE (CFL ~ kappa*|S|^2*dt/dz^2), not advective (CFL ~ |w*|*dt/dz).
    # The advective bolus form was CFL=3.49 in the 5 m surface layer and
    # blew up at step 12; this form is CFL=0.48 there. Survives
    # _compute_tracer_residual (skew flux != kappa_h*lap). Off by default
    # (kappa_gm=0); enabled via dataclasses.replace on coarse runs.
    if p.kappa_gm > 0.0:
        S_x_gm, S_y_gm = _isopycnal_slope(state, p)
        gm_T = _redi_skew_flux_tendency(state.T, S_x_gm, S_y_gm, p,
                                        kappa=p.kappa_gm)
        gm_S = _redi_skew_flux_tendency(state.S, S_x_gm, S_y_gm, p,
                                        kappa=p.kappa_gm)
    else:
        gm_T = 0.0
        gm_S = 0.0

    # Redi isopycnal mixing (skew-flux residual): the DISSIPATIVE counterpart
    # to the GM bolus. Its vertical -κ_redi|S|²∂zC term supplies the APE sink
    # that pure advective bolus lacks (arrests the w* steepening feedback).
    # Reuses the same slopes; off by default (kappa_redi=0).
    if p.kappa_redi > 0.0:
        # Compute slopes once; _gm_bolus_velocity already computed its own,
        # but recomputing here keeps Redi independent of the GM gate (Redi can
        # run with kappa_gm=0 if desired). Cost is minor (3 derivatives).
        S_x, S_y = _isopycnal_slope(state, p)
        redi_T = _redi_skew_flux_tendency(state.T, S_x, S_y, p)
        redi_S = _redi_skew_flux_tendency(state.S, S_x, S_y, p)
    else:
        redi_T = 0.0
        redi_S = 0.0

    dTdt = adv_T + diff_h_T + diff_v_T + heat_T + bulk_T + conv_T + gm_T + redi_T
    dSdt = adv_S + diff_h_S + diff_v_S + conv_S + gm_S + redi_S + rest_S
    # Land: tracers held (no tendency over land).
    dTdt = dTdt * p.wet_mask_z
    dSdt = dSdt * p.wet_mask_z
    return dTdt, dSdt


# ── Linear half-step (FD: explicit diffusion + exact Coriolis + free surface) ─

def _explicit_diffusion_step(u, p, nu, dt_half):
    """Explicit FD horizontal+vertical diffusion over dt_half.

    CFL: nu*dt/dx². At 1° (dx~111km), nu_h=100 -> ~4.6e-5 << 0.25 (safe).
    Biharmonic nu_bi: nu_bi*dt/dx⁴ may exceed the explicit limit at 1°;
    if so it is applied here and must be re-calibrated (see make_solver_global).
    """
    diff = nu * (_laplacian_h(u, p) + _d2_dz2(u, p) * 0.0)  # horiz only here
    # vertical diffusion handled separately for clarity
    return u + diff * dt_half


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


def _free_surface_step_fd(eta, u, v, p, F_rho_x=None, F_rho_y=None, dt_half=None):
    """Forward-backward (Sielecki) free-surface (shallow water) step on lat-lon FD.

      eta^{n+1} = eta^n - dt*H_sw*div_h(ubt^n)            # eta from OLD velocity
      ubt^{n+1} = (ubt^n + dt*(-g*grad_h(eta^{n+1}) + F)) / (1 + r_bt*dt)   # u from NEW eta

    with implicit linear barotropic bottom drag (unconditionally stable, no CFL).
    The spectral solver solved the linear SW exactly per wavenumber (matrix
    exponential, energy-neutral). The FD analogue must NOT use forward-forward
    coupling (both from old state): that has |λ| = sqrt(1+(dt*c*k)^2) > 1 for
    ALL k — unconditionally unstable for free gravity waves (a no-wind 1m
    eta-bump grew to 5.5m/day under it). Forward-backward flips the trace to
    2 - dt^2*g*H*k^2 => |λ|=1 (neutral) under CFL<1, the standard OGCM
    discretization; bottom drag then decays the free mode (|λ|~0.97/step).
    CFL: dt < dx/sqrt(g*H) ~ 85-560s; dt=60 is safe on the global 1° grid
    (basin+meso modes CFL<1; grid-scale handled by Laplacian + polar cap).

    mode_split=True: called n_subcyc times per baroclinic step with dt_bt
    (one forward-backward pair per call = dt_bt of evolution; caller passes
    dt_half=dt_bt). The 3D<->barotropic projection stays in the CALLER: inside
    the subcycle we work on (eta, ubt, vbt) 2D state only, seeded from one
    _barotropic_velocity call at subcycle start.

    NOTE (nu_h placement): the subcycle carries NO nu_h dissipation. The
    baroclinic shear (which nu_h is really there to damp) is invisible to the
    depth-averaged BT state; putting nu_h here damped only the depth-mean
    while the 3D shear grew unchecked (real-grid +0.65 m/s per step at a
    trench node -> tracer-advection blow-up by step 15). nu_h instead runs in
    the L half-steps on the FULL 3D field with an internal subcycle
    (nu_sub_cyc substeps of dt/nu_sub_cyc each), which satisfies the explicit
    diffusion CFL while acting on the shear.
    """
    if dt_half is None:
        dt_half = p.dt / 2.0
    if p.mode_split:
        ubt, vbt = u, v   # subcycle mode: caller passes BT velocity directly
    else:
        ubt, vbt = _barotropic_velocity(u, v, p)

    F_x = jnp.zeros_like(ubt)
    F_y = jnp.zeros_like(vbt)
    if F_rho_x is not None:
        F_x = F_x + F_rho_x
        F_y = F_y + F_rho_y
    F_x = F_x + p.tau_x_2d / (RHO_0 * p.H_sw)
    F_y = F_y + p.tau_y_2d / (RHO_0 * p.H_sw)

    r_bt = p.r_bot if p.bottom_friction == 'linear' else 0.0
    drag = 1.0 / (1.0 + r_bt * dt_half)
    # Mass-conserving (flux-form) divergence: face fluxes zeroed at wet/dry
    # interfaces so the divergence telescopes to zero over the wet domain.
    # The centered (roll) divergence leaks volume at coastlines + closed walls
    # (its stencil reaches into land zeros; the post-update *=wet_mask then
    # discards the compensating volume). Mask ubt/vbt to wet first so the
    # flux-form face averages don't carry land values into open faces.
    div_bt = _divergence_conservative(ubt * p.wet_mask, vbt * p.wet_mask, p)

    # Forward-backward (Sielecki) free-surface coupling: update eta FIRST
    # (old velocity), then update barotropic momentum using the NEW eta
    # gradient. The previous forward-forward coupling (both from old state)
    # has amplification |λ| = sqrt(1 + (dt*c*k)^2) > 1 for ALL k — i.e. it is
    # UNCONDITIONALLY UNSTABLE for free gravity waves (CFL does not save
    # forward-Euler; it only saves centered/leapfrog schemes). At dt=60 the
    # basin-scale seiche grows ~1.0023/step => ~27x/day, which a no-wind
    # 1m eta-bump test confirmed (1m -> 5.5m in 1 day, mean~0 so mass
    # conserved but amplitude growing). Forward-backward flips the trace of
    # the amplification matrix to 2 - dt^2*g*H*k^2, giving |λ| = 1 (neutral)
    # under CFL < 1 — the standard OGCM discretization (MOM6/ROMS/NEMO).
    # Bottom drag then actively decays the free mode (|λ| ~ 0.97/step here),
    # so a perturbed eta relaxes to the steady wind-driven setup instead of
    # amplifying. No iterative solve needed (unlike semi-implicit Helmholtz).
    eta_new = eta - dt_half * p.H_sw * div_bt

    eta_new = eta_new * p.wet_mask
    # Lateral sponge on eta (2D). No-op when sponge_rate_2d == 0.
    sw_decay = jnp.exp(-p.sponge_rate_2d * dt_half)
    #
    # MASS-CONSERVING SPONGE (defect #1 fix): the bare decay eta*=sw_decay
    # changes global volume by dV = sum(A*eta*(sw_decay-1)) over the band. The
    # wind setup makes the band-mean eta NEGATIVE on both hemispheres
    # (subpolar lows / ACC south-of-westerlies minimum), so the sponge was
    # ADDING volume every step -> +0.377 m global mean-eta drift over 90d
    # (steady ~10000 km^3/5d from d5, matches the eta-decay leak budget).
    # Correct by adding the removed volume back UNIFORMLY over the wet domain:
    # total volume exactly conserved, local anomaly damping unchanged, and a
    # uniform eta offset has zero PGF so the dynamics are untouched.
    eta_pre_sponge = eta_new
    eta_new = eta_new * sw_decay
    area_cell = p.dx_2d * p.dy                       # (nx, ny) cell area
    dV_sponge = jnp.sum(area_cell * (eta_new - eta_pre_sponge))
    area_ocean = jnp.maximum(jnp.sum(area_cell * p.wet_mask), 1.0)
    eta_new = eta_new + (-dV_sponge / area_ocean) * p.wet_mask

    # ── Semi-enclosed-sea eta relaxation (Mediterranean artifact fix) ──
    # Gibraltar (14 km wide) is sub-grid on a 1° mesh: the one-cell strait
    # cannot support the observed two-layer exchange, and the residual
    # pressure mismatch drives a spurious NET outflow that drains the basin
    # linearly (~+0.026 m/d, max|eta| 9.5 m at d365 in gpu365_glap —
    # would trip the 15 m watchdog at ~d500). Not an instability: a
    # coarse-grid semi-enclosed-sea artifact that must be BOUNDED for
    # multi-year integrations. Standard OGCM remedy (MOM-family): Rayleigh
    # relax eta toward the basin equilibrium INSIDE the sea only:
    #     eta *= exp(-eta_relax_rate * dt_half)   where eta_relax_mask = 1
    # (eta==0 is the correct equilibrium: the basin-mean PGF at Gibraltar
    # then matches the Atlantic open boundary at the same latitude).
    # MASS-CONSERVING COMPENSATION: the relaxation removes volume
    # dV = sum(A*eta*(decay-1)) over the mask; the SAME uniform-refill trick
    # as the sponge returns it over the global wet domain. A uniform eta
    # offset has zero PGF, so the dynamics are untouched — the relaxation
    # damps the basin anomaly, not the basin water mass.
    eta_relax_decay = jnp.exp(-p.eta_relax_rate * dt_half * p.eta_relax_mask)
    eta_pre_relax = eta_new
    eta_new = eta_new * eta_relax_decay
    dV_relax = jnp.sum(area_cell * (eta_new - eta_pre_relax))
    eta_new = eta_new + (-dV_relax / area_ocean) * p.wet_mask

    # Polar-cap filter: zonally average the poleward rows to kill the
    # cos(lat)->0 metric singularity (dx->0 makes the explicit SW CFL
    # unattainable at the edge). Replaces the unresolved polar dynamics with a
    # zonally-uniform cap value — standard for lat-lon FD OGCMs.
    #
    # CRITICAL: average over WET points only and write back to WET points only.
    # The polar rows are ~30% land (coastlines at high lat). A naive all-column
    # mean mixes ocean (eta != 0) with land (eta == 0), forcing a zonally-
    # uniform value that creates a spurious PGF at EVERY coastline point in the
    # band -> wind-driven barotropic energy injection exactly there. This was
    # the G2 wind-forced blow-up nucleation (max|u| diverged at (79.5, 124.5),
    # a wet point flanked by land, sub-inertial, CFL-safe — not a CFL failure).
    def _cap(field2d):
        """Tapered zonal-mean polar cap on a 2D field (eta, ubt, vbt).

        Blends the wet-point zonal mean into the poleward rows with the same
        cos^2 taper as the 3D cap (_polar_cap_weights) so there is no
        meridional cliff at the cap edge (the hard-cutoff G3 step-12
        blow-up). Land stays at its masked value (0).
        """
        nc = p.polar_cap_rows
        if nc <= 0:
            return field2d
        nt = p.polar_cap_taper
        wts = _polar_cap_weights(nc, nt)               # (nc+nt,)
        # _polar_cap_weights builds jnp.ones/linspace at runtime, which are
        # float64 under jax_enable_x64 — cast to the field's dtype so the
        # blend does not silently promote eta/ubt/vbt to f64.
        wts = wts.astype(field2d.dtype)
        nb = nc + nt
        wts_b = wts.reshape(1, nb)

        def _band(field, wm_band):
            s = field * wm_band                        # (nx, nb)
            wsum = jnp.maximum(jnp.sum(wm_band, axis=0, keepdims=True), 1.0)  # (1,nb)
            zmean = jnp.sum(s, axis=0, keepdims=True) / wsum
            zmean = jnp.broadcast_to(zmean, (p.nx, nb)) * wm_band
            return wts_b * zmean + (1.0 - wts_b) * (field * wm_band)

        south = _band(field2d[:, :nb], p.wet_mask[:, :nb])
        north = _band(field2d[:, -nb:], p.wet_mask[:, -nb:])
        return jnp.concatenate([south, field2d[:, nb:-nb], north], axis=1)

    # CONSISTENT-TRIPLE CAP (god 02-55 #1): cap eta FIRST, then drive the
    # barotropic momentum update from the CAPPED eta's pressure gradient, then
    # cap the resulting ubt/vbt. Previously eta/ubt/vbt were capped to THREE
    # INDEPENDENT zonal means, so in the cap band eta was zonally uniform but
    # ubt/vbt carried a different zonal structure -> the next step's div(ubt)
    # and grad(eta) were dynamically inconsistent -> spurious PGF -> free-mode
    # energy injection (cap-ON NaN step 534 vs cap-OFF 800). Capping eta first
    # and computing the PGF from the capped eta makes ubt/vbt consistent with
    # eta by construction; their cap is then a CFL-safety smoothing, not an
    # independent forcing.
    eta_new = _cap(eta_new)
    # Energy-consistent PGF: the exact adjoint of the conservative divergence
    # (area-weighted), so the FB pair is neutral on the masked non-uniform grid.
    # The centered _d_dx/_d_dy gradient is NOT the adjoint and injects energy.
    grad_eta_x, grad_eta_y = _gradient_conservative(eta_new, p)
    # Barotropic momentum with PGF + forcing (intermediate star state).
    u_star = ubt + dt_half * (-G_EARTH * grad_eta_x + F_x)
    v_star = vbt + dt_half * (-G_EARTH * grad_eta_y + F_y)
    # Semi-implicit barotropic Coriolis. The shallow-water momentum eqn is
    #   du/dt - f*v = -g*grad(eta) + F ;  dv/dt + f*u = -g*grad(eta) + F
    # Treating Coriolis implicitly (unconditionally stable, energy-neutral):
    #   (u_new - u_star)/dt =  +f*v_new
    #   (v_new - v_star)/dt =  -f*u_new
    # => u_new = (u_star + f*dt*v_star) / (1+(f*dt)^2)
    # => v_new = (v_star - f*dt*u_star) / (1+(f*dt)^2)
    # Without this, the barotropic PGF has no geostrophic balance: F_rho (~4e-4
    # m/s^2) drives a convergent ubt that grows eta monotonically (Coriolis was
    # 180x too small) => exponential eta/ubt growth => advection overshoot => NaN.
    # This is the barotropic analogue of the 3D rotation in _linear_half_step.
    fd = p.f * dt_half                 # (nx, ny)
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
    u_new = u + delta_ubt
    v_new = v + delta_vbt
    # No-flux wall: enforce zero normal velocity at the N/S boundary rows on
    # the projected 3D v too, consistent with _step_impl's final mask.
    v_new = v_new * p.interior_mask_z
    return eta_new, u_new, v_new


def _linear_half_step(state, p, dt_half):
    """Linear half-step: FD diffusion + 2D Coriolis + EXPLICIT free surface.

    Unlike the spectral solver (exact spectral diffusion decay + matrix-exp
    free surface), the FD linear step is:
      - explicit horizontal+vertical diffusion (CFL-safe at 1°),
      - exact per-gridpoint Coriolis rotation on the 2D f-field,
      - EXPLICIT forward-Euler free surface (CFL: dt < dx/sqrt(gH) ~ 85-560s;
        dt=60 is safe across the global grid).
    """
    # Explicit diffusion (horizontal Laplacian + vertical d2/dz2).
    # nu_h (momentum Laplacian) CANNOT be applied at dt_half=1800 s in one
    # explicit shot (nu_h*dt_half/dy² = 0.73 > 0.25), but it MUST act on the
    # full 3D baroclinic velocity: the N-step residual subtracts nu_h*lap from
    # the full tendency, so if the L step re-adds nothing the baroclinic SHEAR
    # (invisible to the barotropic subcycle's depth-mean) loses all its
    # damping — the real-grid split run grew +0.65 m/s/step at a trench node
    # (154.5E, 11.5S, k=12) and blew up via tracer advection by step 15.
    # Fix: apply nu_h here subcycled (nu_sub_cyc steps of dt_half/nu_sub_cyc
    # each, CFL 0.06 per substep) — same fixed-operator subcycling pattern as
    # conv_nsub. Monolithic dt=60-300 keeps the single-shot form (CFL-safe).
    if p.mode_split and p.n_subcyc > 0:
        # subcycle count sized by the 0.5*LHS FTCS criterion. Legacy sizing
        # (nu_nsub=None) reuses n_subcyc (24 substeps at dt=3600/dt_bt=150,
        # LHS 0.136 — a 3.7x stability margin). nu_nsub right-sizes it from
        # the actual metric worst-case: nu_h*dt/(n*dy_min^2) <= 0.5 (see
        # make_solver_global) — ~10 substeps at the same margin, 2.4x fewer
        # laplacian pairs per L half-step.
        u, v = state.u, state.v
        n_nu = max(1, int(p.n_subcyc if p.nu_nsub is None else p.nu_nsub))
        dts = dt_half / n_nu
        if p.use_scan:
            # Fused nu_h subcycle: n_nu laplacian-pair applications.
            def _nu_sub(carry, _):
                cu, cv = carry
                return (cu + p.nu_h * _laplacian_h(cu, p) * dts,
                        cv + p.nu_h * _laplacian_h(cv, p) * dts), None

            (u, v), _ = jax.lax.scan(_nu_sub, (u, v), None, length=n_nu)
        else:
            for _ in range(n_nu):
                u = u + p.nu_h * _laplacian_h(u, p) * dts
                v = v + p.nu_h * _laplacian_h(v, p) * dts
    else:
        u = state.u + p.nu_h * _laplacian_h(state.u, p) * dt_half
        v = state.v + p.nu_h * _laplacian_h(state.v, p) * dt_half
    T = state.T + p.kappa_h * _laplacian_h(state.T, p) * dt_half
    S = state.S + p.kappa_h * _laplacian_h(state.S, p) * dt_half
    # Scale-selective biharmonic (∇⁴): damps grid-scale modes far more than
    # large-scale. Spectral solver applied this via exp(-nu_bi*k⁴*dt); the FD
    # analogue is an explicit forward-Euler step. CFL: nu_bi*dt/dx⁴ < ~0.05.
    # nu_bi is re-calibrated for 1° (much larger than the spectral 1e12, which
    # was tuned for the regional 0.1° grid where dx⁴ is ~1600x smaller).
    if p.nu_bi > 0.0:
        u = u - p.nu_bi * _biharmonic_h(state.u, p) * dt_half
        v = v - p.nu_bi * _biharmonic_h(state.v, p) * dt_half
    if p.kappa_bi > 0.0:
        T = T - p.kappa_bi * _biharmonic_h(state.T, p) * dt_half
        S = S - p.kappa_bi * _biharmonic_h(state.S, p) * dt_half
    u = u + p.nu_v * _d2_dz2(state.u, p) * dt_half
    v = v + p.nu_v * _d2_dz2(state.v, p) * dt_half
    T = T + _diff_v_flux_tendency(state.T, p.kappa_v, p) * dt_half
    S = S + _diff_v_flux_tendency(state.S, p.kappa_v, p) * dt_half
    # Mask: no diffusion updates over land or below seafloor (ghost water).
    # Hold land/ghost values at their ORIGINAL state (not zero): masking to
    # zero creates a T=0 cliff at every coastline that the (unmasked)
    # _laplacian_h in the N-step tracer residual reads as a huge gradient,
    # seeding a spurious baroclinic PGF that grows exponentially. Holding the
    # pre-step value preserves the no-flux (flat) land value the diffusive
    # stencil already assumes (mirror ghost = boundary value => no cliff).
    u = u * p.wet_mask_z + state.u * (1.0 - p.wet_mask_z)
    v = v * p.wet_mask_z + state.v * (1.0 - p.wet_mask_z)
    T = T * p.wet_mask_z + state.T * (1.0 - p.wet_mask_z)
    S = S * p.wet_mask_z + state.S * (1.0 - p.wet_mask_z)

    # Lateral sponge (Rayleigh damping) — exponential decay, unconditional
    # (no damping CFL). Applied in the linear half-step so Strang splitting
    # gives total sponge time = dt per full step. Absorbs the wind-driven
    # barotropic energy that Laplacian dissipation can't arrest within its
    # CFL cap, which otherwise piles up at the polar edge rows. No-op when
    # sponge_rate == 0 (decay == 1, T_clim == 0 -> T unchanged).
    decay = jnp.exp(-p.sponge_rate * dt_half)            # (nx,ny,1)
    u = u * decay
    v = v * decay
    T = p.T_clim_3d + (T - p.T_clim_3d) * decay
    S = p.S_clim_3d + (S - p.S_clim_3d) * decay

    # Coriolis rotation (2D f-field, exact)
    u, v = _coriolis_rotation_2d(u, v, p.f, dt_half)

    # Semi-implicit free surface (with density barotropic PGF).
    # mode_split: the free surface runs ONCE per baroclinic step in
    # _step_impl's barotropic subcycle (dt_bt), not here — the external-wave
    # CFL forbids dt_half=1800 s here. The linear step keeps diffusion
    # (nu_h subcycled above) + sponge + Coriolis only.
    if p.mode_split:
        return JaxStateG(u, v, T, S, state.eta)
    F_rho_x, F_rho_y = _compute_bt_rho_pgf(state, p)
    eta, u, v = _free_surface_step_fd(state.eta, u, v, p, F_rho_x, F_rho_y, dt_half)
    return JaxStateG(u, v, T, S, eta)


# ── Nonlinear explicit step (forward-backward RK2, FD) ─────────────

def _compute_tracer_residual(state, p):
    """Tracer tendency minus the linear diffusion (handled by linear step).

    The Strang split L(dt/2)·N(dt)·L(dt/2) handles ALL linear dissipation
    (Laplacian + biharmonic) in the L step. The residual (N part) must
    therefore SUBTRACT the linear dissipation that the full tendency
    included, so it is not double-applied. For the Laplacian this is
    dTdt -= kappa_h*lap (the tendency's +kappa_h*lap is removed; L re-adds
    it). The biharmonic is NOT in _compute_tracer_tendency (it is an L-only
    term), so it must NOT appear in the residual at all — a +kappa_bi*biharm
    here would be re-applied by N(dt) and cancel the L-step damping
    (-dt/2 + dt - dt/2 = 0 net), making biharmonic a silent no-op.
    """
    dTdt, dSdt = _compute_tracer_tendency(state, p)
    dTdt = dTdt - p.kappa_h * _laplacian_h(state.T, p)
    dSdt = dSdt - p.kappa_h * _laplacian_h(state.S, p)
    dTdt = dTdt - p.kappa_v * _d2_dz2(state.T, p)
    dSdt = dSdt - p.kappa_v * _d2_dz2(state.S, p)
    return dTdt, dSdt


def _compute_momentum_residual(state, p):
    """Momentum tendency minus linear parts (diffusion, Coriolis, bt PGF, bt wind).

    Biharmonic hyperviscosity is an L-only term (not in the full momentum
    tendency), so it must NOT appear here — see _compute_tracer_residual
    for the no-op cancellation argument.
    """
    dudt, dvdt = _compute_momentum_tendency(state, p)
    # nu_h subtraction is unconditional: the full tendency ALWAYS includes
    # diff_h, so the residual must ALWAYS remove it — the L step re-adds it
    # (monolithic single-shot, split subcycled over n_subcyc substeps).
    # Skipping the subtraction in split mode (an earlier draft's
    # `if not p.mode_split` guard) applied nu_h inside the N-step at dt=3600,
    # where nu_h*dt/dy^2 = 1.46 >> 0.25 exceeds the explicit-diffusion CFL.
    dudt = dudt - p.nu_h * _laplacian_h(state.u, p)
    dvdt = dvdt - p.nu_h * _laplacian_h(state.v, p)
    dudt = dudt - p.f[:, :, None] * state.v
    dvdt = dvdt + p.f[:, :, None] * state.u
    # barotropic PGF from eta. The full tendency's 3D PGF contains the eta
    # part via p_bt = RHO_0*G_EARTH*eta (top-node pressure,
    # _compute_hydrostatic_pressure) differentiated by
    # _gradient_conservative_3d with PER-LAYER wet/ghost face gating.
    # mode_split: cancel it with the SAME operator on the SAME field, so the
    # subtraction is EXACT and the external mode acts on 3D momentum solely
    # through the subcycle projection (a bare or column-gated 2D gradient
    # leaves a spurious eta-PGF at every face where the layer gate differs
    # from the column gate, scaling with dt).
    # Monolithic: keep the historical bare _d_dx/_d_dy subtraction bit-exact
    # to the pre-split solver (all production baselines, dt=60-300 runs) —
    # its operator mismatch is O(coastline metric) and was stable there.
    if p.mode_split:
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
    # _compute_momentum_tendency returns dudt already masked to 0 in ghost
    # water (layers below the seafloor, wet_mask_z==0), but the barotropic
    # subtractions above (bt_pgf, bt_rho_pgf, bt_wind) are broadcast uniformly
    # over ALL depths — so in ghost-water layers the residual was -bt_pgf
    # (a spurious PGF) instead of 0, mismatching the masked tendency there.
    # Masking closes the mismatch so the residual equals the (masked) full
    # tendency minus its linear parts everywhere, including ghost water. This
    # keeps the N-step residual clean in ghost layers (verified ~1e-21, machine
    # zero) so no spurious tendency leaks into the wet column via the vertical
    # operators that couple adjacent layers.
    dudt = dudt * p.wet_mask_z
    dvdt = dvdt * p.wet_mask_z
    return dudt, dvdt


def _explicit_full_step(state, p, dt):
    """Forward-backward RK2 for nonlinear tendencies (FD).

    Tracers updated first (old velocity), then momentum uses predicted T
    for the baroclinic PGF — shifts internal-wave eigenvalues left of the
    imaginary axis for neutral stability. Same structure as spectral solver.
    """
    dT1, dS1 = _compute_tracer_residual(state, p)
    T_pred = state.T + dT1 * dt
    S_pred = state.S + dS1 * dt
    state_T = JaxStateG(state.u, state.v, T_pred, S_pred, state.eta)

    du1, dv1 = _compute_momentum_residual(state_T, p)
    # ── Semi-implicit linear bottom drag (split mode only) ──
    # The residual's -r_bot*u is forward-Euler here: pure-drag amplitude
    # 1 - r*dt flips past the stability bound |1-r*dt|<=1 (r*dt > 2) and the
    # RK2 form 1 - rdt + (rdt)^2/2 past r*dt = 2. At the historical dt=60-300
    # (r*dt = 0.06-0.3) explicit was fine; the mode-split dt=3600 gives
    # r*dt = 3.6 -> the bottom layer amplifies ~2.6x per step with sign
    # alternation (smoke-test growth 3x/step at the bottom-layer node
    # (8,30,7), the dominant injector after the FS/nu_h/eta-PGF fixes).
    # Both RK2 stages evaluate the residual at the OLD velocity
    # (state_T / state_T_new carry state.u, state.v), so the drag part is
    # identical in du1/du2: strip it from both stages and apply the EXACT
    # linear-drag decay exp(-r_bot*dt) after the RK2 update — unconditionally
    # stable, exact for the pure-drag sub-operator, and the FD analogue of
    # the spectral solver's exponential linear dissipation. Monolithic path
    # is untouched (bit-exact to the pre-split solver; its dt range keeps
    # the explicit form stable).
    implicit_drag = p.mode_split and p.bottom_friction == 'linear'
    if implicit_drag:
        bm = p.bottom_mask * p.wet_mask_z
        du1 = du1 + p.r_bot * state.u * bm
        dv1 = dv1 + p.r_bot * state.v * bm
    u_pred = state.u + du1 * dt
    v_pred = state.v + dv1 * dt
    state_pred = JaxStateG(u_pred, v_pred, T_pred, S_pred, state.eta)

    dT2, dS2 = _compute_tracer_residual(state_pred, p)
    T_new = state.T + 0.5 * (dT1 + dT2) * dt
    S_new = state.S + 0.5 * (dS1 + dS2) * dt
    state_T_new = JaxStateG(state.u, state.v, T_new, S_new, state.eta)

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
    return JaxStateG(u_new, v_new, T_new, S_new, state.eta)


def _polar_cap_weights(ncap, ntaper):
    """Blend weights for the tapered polar cap (south end; north is mirrored).

    Returns a 1D array of length ncap+ntaper: the blend fraction of the
    zonal mean applied to each row, from the pole inward. Rows [0:ncap] get
    weight 1.0 (full zonal average — kills the cos(lat)->0 metric
    singularity); rows [ncap:ncap+ntaper] ramp 1->0 via a cos^2 taper so
    there is no meridional discontinuity at the cap's inner edge. A hard
    cutoff (ntaper=0) creates a cliff at j=ncap: the cap forces j<ncap to a
    zonally-uniform value inconsistent with the free row j=ncap, and _d_dy
    amplifies that cliff exponentially (the G3 step-12 pole-wall blow-up).
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


def _apply_polar_cap_3d(field3d, p):
    """Tapered zonal-average polar cap on a 3D field.

    Zonally averages the poleward rows (wet-point mean) and blends the
    result into the field with a cos^2 taper across `polar_cap_taper` extra
    transition rows, so the cap edge is smooth (no meridional cliff). See
    _polar_cap_weights for the hard-edge failure mode this replaces. The
    standard lat-lon FD OGCM treatment (MOM6 polar cap / Arctic fold).
    """
    ncap = p.polar_cap_rows
    if ncap <= 0:
        return field3d
    ntaper = p.polar_cap_taper
    # 3D wet mask (1 = water at THIS depth, 0 = land OR below seafloor).
    # The 2D wet_mask is column-wide: using it here let the ghost nodes'
    # T_ref fill (15.0 C, config.T_ref) below shallow-seafloor columns
    # enter the zonal mean at deep levels. With polar_cap_rows=2 at 4000 m
    # ~1/3 of the band columns are ghost there, dragging the cap T to
    # ~+15 C — a meridional cliff against the -0.3 C WOA deep T that
    # seeded the Southern-Ocean cold-pole blow-up (d170 collapse).
    wts = _polar_cap_weights(ncap, ntaper)       # (ncap+ntaper,)
    # Runtime-built jnp.ones/linspace are float64 under jax_enable_x64; cast
    # to the field's dtype or the blend promotes the whole field to f64
    # (silently defeats the fp32 state/params cast downstream).
    wts = wts.astype(field3d.dtype)
    nb = ncap + ntaper
    wmz = p.wet_mask_z                           # (nx, ny, nz), 3D mask

    def _cap_band(field, wm_band):
        # Wet-point zonal mean over the FULL band (one value per (row,z)),
        # broadcast back; land/ghost stays at its masked value.
        s = field * wm_band
        w = jnp.maximum(wm_band, 1e-12)
        wsum = jnp.sum(w, axis=0, keepdims=True)   # (1, nb, nz)
        zmean = jnp.sum(s, axis=0, keepdims=True) / wsum
        zmean = jnp.broadcast_to(zmean, field.shape) * wm_band
        # Blend: wts[r] * zmean_row + (1-wts[r]) * field_row.
        wts_b = wts.reshape(1, nb, 1)
        return wts_b * zmean + (1.0 - wts_b) * (field * wm_band)

    south = _cap_band(field3d[:, :nb], wmz[:, :nb])
    north = _cap_band(field3d[:, -nb:], wmz[:, -nb:])
    return jnp.concatenate([south, field3d[:, nb:-nb], north], axis=1)


def _polar_cap_3d(field3d, p):
    """Tapered zonal-average polar cap on a 3D field (u, v, T, S).

    Thin wrapper around _apply_polar_cap_3d; kept for call-site clarity.
    See _apply_polar_cap_3d / _polar_cap_weights for the taper rationale.
    """
    return _apply_polar_cap_3d(field3d, p)


def _step_impl(state, p):
    """Strang splitting: L(dt/2) -> N(dt) -> L(dt/2).

    mode_split=True: identical L/N/L baroclinic core, but the linear half-
    steps carry only diffusion/sponge/Coriolis (no free surface, no nu_h).
    After the second L half-step, n_subcyc barotropic forward-backward
    subcycles of dt_bt each (exact fill of p.dt) evolve (eta, ubt, vbt),
    driven by the rho-PGF + wind forcing computed from the baroclinic state
    at step start (MOM-style coupling lag), with nu_h dissipation inside
    each subcycle (nu_h CFL is satisfied at dt_bt, not at dt). The final
    (ubt, vbt) is projected back onto the 3D velocity as a uniform-in-depth
    delta — same projection the monolithic path applies every step, just
    once per baroclinic step instead of twice.
    """
    dt_half = p.dt / 2.0
    # Capture land/ghost values BEFORE the step. Final masking holds these
    # (not zero): masking land T to 0 builds a coastline cliff that the
    # unmasked _laplacian_h in the next step's tracer residual reads as a
    # huge gradient, seeding an exponentially-growing spurious PGF.
    land_u, land_v = state.u, state.v
    land_T, land_S = state.T, state.S
    state = _linear_half_step(state, p, dt_half)
    state = _explicit_full_step(state, p, p.dt)
    state = _linear_half_step(state, p, dt_half)

    # ── mode split: barotropic subcycle (free surface + nu_h) ──
    # Runs AFTER the L/N/L core on the un-masked final 3D state (before the
    # polar cap / land hold below, so the subcycle sees the same masked,
    # capped input the monolithic path fed _free_surface_step_fd).
    if p.mode_split:
        # Coupling forcing from the baroclinic state, held fixed over the
        # subcycle (MOM-style forcing lag at dt=3600 s is standard).
        F_rho_x, F_rho_y = _compute_bt_rho_pgf(state, p)
        ubt0, vbt0 = _barotropic_velocity(state.u, state.v, p)
        ubt, vbt = ubt0, vbt0
        eta = state.eta

        # lax.scan barotropic subcycle (when use_scan): one fused device-side
        # loop of n_subcyc forward-backward pairs instead of n_subcyc unrolled
        # copies of the same 5-kernel block. Numerically IDENTICAL (same
        # _free_surface_step_fd called n_subcyc times) — scan just compiles
        # the body once, cutting XLA graph size and host launch overhead.
        # Python-loop path stays default (None) = bit-exact legacy trace.
        if p.use_scan:
            def _fb(carry, _):
                eta_i, ubt_i, vbt_i = carry
                eta_i, ubt_i, vbt_i = _free_surface_step_fd(
                    eta_i, ubt_i, vbt_i, p, F_rho_x, F_rho_y, dt_half=p.dt_bt)
                return (eta_i, ubt_i, vbt_i), None

            (eta, ubt, vbt), _ = jax.lax.scan(
                _fb, (eta, ubt, vbt), None, length=int(p.n_subcyc))
        else:
            for _ in range(int(p.n_subcyc)):
                # One FB pair per call = dt_bt of evolution (caller passes
                # dt_half=dt_bt). ubt/vbt seeded once; each subcycle feeds the
                # 2D (eta, ubt, vbt) triple forward.
                eta, ubt, vbt = _free_surface_step_fd(
                    eta, ubt, vbt, p, F_rho_x, F_rho_y, dt_half=p.dt_bt)
        # Project the subcycle's net barotropic delta onto the 3D velocity
        # (uniform over depth) — same projection as the monolithic path.
        delta_ubt = (ubt - ubt0)[:, :, None]
        delta_vbt = (vbt - vbt0)[:, :, None]
        state = JaxStateG(state.u + delta_ubt, state.v + delta_vbt,
                          state.T, state.S, eta)

    # 3D polar-cap filter: zonally average the cap rows of u,v,T,S to kill
    # the cos(lat)->0 metric singularity in the diffusion/advection operators.
    u = _polar_cap_3d(state.u, p)
    v = _polar_cap_3d(state.v, p)
    T = _polar_cap_3d(state.T, p)
    S = _polar_cap_3d(state.S, p)
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
    return JaxStateG(u, v, T, S, state.eta)


# ── Public API ──────────────────────────────────────────────────────

def make_solver_global(grid, physics, dt, forcing=None, eos_type='linear',
                       T_atm=None, lambda_bulk=0.0,
                       S_ref_surf=None, sss_restore_days=0.0,
                       sponge_days=0.0, sponge_cells=0,
                       T_init=None, S_init=None,
                       polar_cap_rows=2, polar_cap_taper=3, return_params=False,
                       eta_relax_days=0.0, eta_relax_box=None,
                       eta_relax_buffer=1.0, dynamic_forcing=False,
                       mode_split=False, dt_bt=150.0, nu_nsub=None,
                       dtype='float64', use_scan=False):
    """Create a JIT-compiled global FD ocean solver.

    Args mirror the spectral make_solver where applicable. Key differences:
      - No spectral wavenumbers/decay factors (FD operators instead).
      - Forcing is physical-space 2D (tau_x, tau_y, Q_heat); no pre-FFT.
      - Free surface is forward-backward (Sielecki) + polar-cap filter;
        requires dt below the external-gravity-wave CFL.
      - Lateral sponge at the POLAR EDGE rows (not N/S periodic boundaries):
        the global polar edge is the analogue of the regional N/S boundary.
        Wind-driven barotropic energy piles up there (Laplacian can't arrest
        it within its CFL cap); the sponge absorbs it. cosine-tapered.
      - No SST restore (bulk flux only). Surface SALINITY restoring
        (S_ref_surf + sss_restore_days>0) is available — Haney relaxation
        of SSS to climatology, mirroring the bulk-heat-flux form.
      - Optional eta_relax: Rayleigh SSH relaxation inside a semi-enclosed
        sea (mass-conserving; bounds the sub-grid-strait drainage artifact
        for multi-year runs).
      - mode_split=True: barotropic subcycling of the free surface (dt_bt)
        so the baroclinic dt can exceed the external-gravity-wave CFL
        (3600 s at 1 deg). Off by default — off is bit-exact to the
        pre-split solver.
      - nu_nsub: None = legacy nu_h subcycle count (n_subcyc); 'cfl' =
        right-sized from the explicit-diffusion CFL; int = verbatim.
      - dtype='float32' casts params/state to fp32 (1:64 FP64:FP32 on Ada
        GPUs — biggest kernel-time lever). Default 'float64' = bit-exact.
      - use_scan=True runs the barotropic subcycle as lax.scan (numerically
        identical, smaller XLA graph / fewer host launches).
    """
    base = make_fd_params(grid)
    nx, ny, nz = base.nx, base.ny, base.nz

    if forcing is None:
        tau_x_2d = jnp.zeros((nx, ny))
        tau_y_2d = jnp.zeros((nx, ny))
        Q_heat_2d = jnp.zeros((nx, ny))
    else:
        tau_x_2d, tau_y_2d, Q_heat_2d = (jnp.array(f) for f in forcing)

    # Effective shallow-water depth = vertical grid span
    H_sw = float(jnp.sum(jnp.array(grid.dz)))
    dz_norm = (jnp.array(grid.dz).reshape(1, 1, -1) / H_sw)

    if T_atm is not None and lambda_bulk > 0.0:
        T_atm_3d = jnp.array(T_atm)[:, :, None]
    else:
        T_atm_3d = jnp.zeros((nx, ny, 1))
        lambda_bulk = 0.0

    # ── Surface salinity restoring (Haney) ──
    # dSdt += -(SSS - S_ref)/tau at wet surface cells. tau=0 → off
    # (bit-exact to the pre-restoring solver). S_ref_surf is a (nx, ny)
    # climatological SSS field (e.g. WOA surface salinity); the restoring
    # is a TRUE salt flux (psu/s, no heat_factor — salinity has no rho*cp).
    if S_ref_surf is not None and sss_restore_days > 0.0:
        S_ref_2d = jnp.array(S_ref_surf)
        restore_coef_S = 1.0 / (sss_restore_days * 86400.0)
    else:
        S_ref_2d = jnp.zeros((nx, ny))
        restore_coef_S = 0.0

    # ── Lateral sponge (polar-edge Rayleigh damping) ──
    # Same construction as the regional spectral solver's N/S sponge, but
    # applied at both lat edges (the polar cap rows). cosine-tapered from
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
        if T_init is not None:
            T_clim_3d = jnp.array(T_init)
            S_clim_3d = jnp.array(S_init) if S_init is not None else jnp.zeros_like(T_clim_3d)
        else:
            T_clim_3d = jnp.zeros((nx, ny, nz))
            S_clim_3d = jnp.zeros((nx, ny, nz))
    else:
        sponge_rate_2d = jnp.zeros((nx, ny))
        sponge_rate = jnp.zeros((nx, ny, 1))
        T_clim_3d = jnp.zeros((nx, ny, nz))
        S_clim_3d = jnp.zeros((nx, ny, nz))

    # ── Semi-enclosed-sea eta relaxation mask (Mediterranean artifact fix) ──
    # A parameterized lat/lon box; 0 rate = off (bit-exact to the old runs).
    # The buffer band (mask tapers 1 -> 0 over `eta_relax_buffer` degrees
    # around the box edge) keeps the PGF smooth at the mask boundary so the
    # relaxation itself cannot seed a new PGF cliff at Gibraltar.
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

    # ── 2/3-rule dealias mask for the periodic lon axis ──
    # Nonlinear advection products (u*du/dx, T*div_h, ...) and the
    # div_h-integrated w all amplify the 2-dx grid-scale mode. The spectral
    # baseline (jax_solver.py) removes it with _dealias_h (2/3 FFT rule) on w
    # AND on every advection tendency (adv_u, adv_v, adv_T); the FD solver
    # does the equivalent via _dealias_h_fd. lon is periodic -> a 2/3 FFT
    # dealias in lon is exact; the closed lat wall can't use FFT, so the lat
    # 2-dx is killed by a 5-pt binomial low-pass in _dealias_h_fd.
    _kmax = nx // 2
    _keep = max(1, int(_kmax * 2 / 3))
    _kidx = np.fft.fftfreq(nx) * nx            # 0..nx/2, -nx/2..-1
    _lon_mask_np = (np.abs(_kidx) <= _keep).astype(np.float64).reshape(nx, 1, 1)
    dealias_lon_mask = jnp.array(np.broadcast_to(_lon_mask_np, (nx, 1, 1)))

    # ── Mode split (baroclinic/barotropic) ──
    # n_subcyc = exact number of barotropic subcycles filling dt: dt_bt_eff
    # = dt/n_subcyc (rounded so each subcycle is a uniform dt_bt). With
    # mode_split=False all fields are inert (n_subcyc=0) and _step_impl
    # takes the bit-exact monolithic path.
    if mode_split:
        n_subcyc = max(1, int(round(dt / dt_bt)))
        dt_bt_eff = dt / n_subcyc
    else:
        n_subcyc = 0
        dt_bt_eff = 0.0

    # ── Compute dtype ──
    # 'float64' (default) = legacy bit-exact path. 'float32' casts every
    # params array and zeros template to fp32 — on Ada-class GPUs (L20X:
    # FP64:FP32 = 1:64) this is the single biggest kernel-time lever. The
    # runner still reads/writes float64 npz (I/O casts at the boundary).
    # NOTE: params is built further below; the fp32 cast of params happens
    # right after construction (see "fp32 cast" after FDPhysParams(...)).
    state_dtype = jnp.float32 if dtype == 'float32' else jnp.float64

    # ── nu_h subcycle right-sizing (split L half-steps) ──
    # Legacy (None): n_nu = n_subcyc (24 at dt=3600/dt_bt=150), explicit
    # diffusion CFL LHS = nu_h*dt/(2*n_subcyc*dy^2) ~ 0.136 at nu_h=5e6,
    # dy=111 km — a 3.7x margin under the 0.5 bound. nu_nsub='cfl' sizes
    # from the actual metric worst case with the same 2x safety margin:
    #     n = ceil(nu_h * dt / (0.25 * dy^2))
    # (~10 substeps here — 2.4x fewer laplacian pairs per L half-step). An
    # int is honored verbatim (probe/benchmark override).
    if nu_nsub == 'cfl':
        dy_min = float(np.min(np.asarray(grid.dy)))
        nu_nsub = max(1, int(np.ceil(physics.nu_h * dt / (0.25 * dy_min ** 2))))
    elif nu_nsub is not None:
        nu_nsub = max(1, int(nu_nsub))

    params = FDPhysParams(
        dx_2d=base.dx_2d, dy=base.dy, cos_lat=base.cos_lat,
        inv_dx=base.inv_dx, inv_dy=base.inv_dy,
        inv_dx2=base.inv_dx2, inv_dy2=base.inv_dy2,
        f=base.f, wet_mask=base.wet_mask, wet_mask_3d=base.wet_mask_3d,
        wet_mask_z=base.wet_mask_z,
        interior_mask=base.interior_mask, interior_mask_z=base.interior_mask_z,
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
        eos_type=eos_type, r_bot=physics.r_bot, cd=physics.cd,
        bottom_friction=physics.bottom_friction,
        tau_x_2d=tau_x_2d, tau_y_2d=tau_y_2d, Q_heat_2d=Q_heat_2d,
        H_sw=H_sw, dz_norm=dz_norm, dt=dt,
        T_atm_3d=T_atm_3d, lambda_bulk=lambda_bulk,
        S_ref_2d=S_ref_2d, restore_coef_S=restore_coef_S,
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
        conv_nsub=max(1, int(np.ceil(
            physics.kappa_conv * dt / (0.4 * float(jnp.min(jnp.array(grid.dz))) ** 2)))),
        nu_nsub=nu_nsub,
        use_scan=bool(use_scan),
        # Vertical-advection subcycles: keep dt*(|u|/dx + |v|/dy + w/dz) below
        # ~0.5 per substep. The dominant constraint is w/dz in the 5 m surface
        # layer (equatorial upwelling w~3e-3 m/s -> dt*w/dz ~ 2.2 at dt=3600);
        # horizontal terms (~1e-6/s at |u|~1 m/s) are negligible. Sized from
        # a nominal w_max=4e-3 m/s (upwelling + western-boundary downwelling
        # overshoot), 0.5 target CFL per substep. Monolithic dt=60-300:
        # dt*w/dz <= 0.18 -> adv_nsub=1, path untouched.
        adv_nsub=max(1, int(np.ceil(dt * 4.0e-3 / (0.5 * float(jnp.min(
            jnp.array(grid.dz))))))),
    )

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
    # Same compiled graph as step() except the 2D forcing (tau_x, tau_y,
    # Q_heat) is passed as RUNTIME arguments instead of baked constants.
    # params._replace on the namedtuple swaps the three fields for traced
    # placeholders; XLA compiles ONE graph shared by all 12 monthly
    # snapshots (no 12× memory — the exact failure that crashed the
    # regional 365d seasonal design). With dynamic_forcing=False nothing
    # is built (bit-exact to the pre-dynamic path).
    step_dyn = None
    if dynamic_forcing:
        @jax.jit
        def step_dyn(state, tau_x, tau_y, q_heat):
            return _step_impl(state, params._replace(
                tau_x_2d=tau_x, tau_y_2d=tau_y, Q_heat_2d=q_heat))

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
        return JaxStateG(u, v, T, S, eta)

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


# ── MMS verification (G1) ──────────────────────────────────────────
# Manufactured-solution checks for the FD operators. These are the FD
# analogues of the spectral Tier-1 tests (test_spectral_ops.py). Spectral
# achieved ~1e-17 (machine precision); 2nd-order FD should converge as
# error ∝ dx² (~1e-3 at 1°). This is the EXPECTED and honest FD baseline.

def _mms_run():
    """Run MMS checks on a small synthetic global grid (no ETOPO needed).

    Returns a dict of {test_name: (error, passes_threshold)}.
    Builds a tiny 36×18 grid (10° resolution) so the FD convergence is
    visible and the test is fast. Uses analytic trigonometric fields.
    """
    from config import GlobalGridConfig, R_EARTH
    import numpy as np

    # Tiny global grid for MMS (coarse so dx is large -> FD error visible)
    nx, ny, res = 36, 14, 360.0 / 36   # 10° resolution
    # Build metric fields directly (avoid ETOPO file dependency in unit test)
    lon = res * (0.5 + np.arange(nx))              # 5, 15, ..., 355
    lat = np.linspace(-75.0, 75.0, ny)             # avoid poles (cos->0)
    lon_2d, lat_2d = np.meshgrid(lon, lat, indexing='ij')
    cos_lat = np.cos(np.radians(lat))
    dx_2d = np.broadcast_to(R_EARTH * np.radians(res) * cos_lat, (nx, ny)).copy()
    dy = R_EARTH * np.radians(res)
    f = 2.0 * OMEGA * np.sin(np.radians(lat_2d))
    wet_mask = np.ones((nx, ny))                   # all-ocean for MMS

    z = np.array([0, -10, -30, -60], dtype=np.float64)
    dz = np.abs(np.diff(z))
    nz = len(z)

    # Minimal FDParams (only horizontal fields needed for these tests)
    class _G:
        pass
    g = _G()
    g.dx_2d = dx_2d; g.dy = dy; g.cos_lat = cos_lat
    g.f = f; g.wet_mask = wet_mask; g.wet_mask_3d = wet_mask[..., None]
    g.z = z; g.dz = dz; g.nz = nz
    g.nx = nx; g.ny = ny
    p = make_fd_params(g)

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
    # NB: at 10° resolution with 2 lat-oscillations, 2nd-order FD truncation
    # is ~7% (each wavelength ~5 points). This is EXPECTED FD behavior, not a
    # bug — the _mms_convergence() check proves true 2nd-order convergence.
    # Threshold set generously; the convergence order is the real gate.
    du_dy_fd = np.array(_d_dy(jnp.array(u3d), p))[:, :, 0]
    du_dy_an = (-np.sin(m * lon_rad) * n * np.sin(n * lat_rad)) / R_EARTH
    err = np.sqrt(np.mean((du_dy_fd - du_dy_an) ** 2)) / (np.abs(du_dy_an).max() + 1e-30)
    results['d_dy_rel_L2'] = (float(err), err < 0.10)

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
    from config import R_EARTH
    import numpy as np

    def ddy_error(ny):
        nx, res = 36, 360.0 / 36
        lat = np.linspace(-75.0, 75.0, ny)
        lon_2d, lat_2d = np.meshgrid(res * (0.5 + np.arange(nx)), lat, indexing='ij')
        cos_lat = np.cos(np.radians(lat))
        dx_2d = np.broadcast_to(R_EARTH * np.radians(res) * cos_lat, (nx, ny)).copy()
        dy = R_EARTH * np.radians(2 * 75.0 / (ny - 1))   # actual dy for this ny
        f = 2.0 * OMEGA * np.sin(np.radians(lat_2d))
        wet_mask = np.ones((nx, ny))
        z = np.array([0, -10, -30, -60], dtype=np.float64); dz = np.abs(np.diff(z)); nz = len(z)
        class _G: pass
        g = _G()
        g.dx_2d = dx_2d; g.dy = dy; g.cos_lat = cos_lat; g.f = f
        g.wet_mask = wet_mask; g.wet_mask_3d = wet_mask[..., None]
        g.z = z; g.dz = dz; g.nz = nz; g.nx = nx; g.ny = ny
        p = make_fd_params(g)
        lat_rad = np.radians(lat_2d); lon_rad = np.radians(lon_2d)
        n = 2.0
        u2d = np.sin(2.0 * lon_rad) * np.cos(n * lat_rad)
        u3d = np.broadcast_to(u2d[..., None], (nx, ny, nz)).copy()
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
    print("=== Global FD Solver — G1 MMS verification ===")
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
