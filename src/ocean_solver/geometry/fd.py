"""Mechanically preserved FD geometry implementation."""
from ocean_solver.geometry.columns import nodal_control_thickness
from ocean_solver.numerics.backend import jnp, np
from ocean_solver.state.types import FDParams


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
