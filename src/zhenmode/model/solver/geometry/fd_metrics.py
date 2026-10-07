"""Finite-difference metric arrays and vertical control thicknesses."""

from zhenmode.model.solver.geometry.grid import fixed_reference_nodal_cells, nodal_control_thickness
from zhenmode.model.solver.numerics.backend import jnp, np
from zhenmode.model.solver.numerics.contacts import OFFSETS, shift_vertical
from zhenmode.model.solver.state import FDParams


def make_fd_params(grid, column_geometry='legacy'):
    """Build FDParams from a GlobalOceanGrid (grid.py).

    Precomputes all metric inverse fields so the FD operators are pure
    array arithmetic (no division inside the hot loop).
    """
    if column_geometry not in {'legacy', 'nodal_dual_v1', 'fixed_partial_v1'}:
        raise ValueError("unknown column_geometry")
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    if column_geometry == 'fixed_partial_v1':
        # The volume-weighted contact identities use dx=L*cos_lat and
        # A=dx*dy. Arbitrary independent row metrics cannot use that operator.
        dx, cosine, meridional = (np.asarray(a) for a in (grid.dx_2d, grid.cos_lat, grid.dy))
        if (dx.shape != (nx, ny) or cosine.shape != (ny,) or meridional.ndim != 0
                or any(a.dtype.kind not in 'fiu' for a in (dx, cosine, meridional))
                or any(not np.isfinite(a).all() or np.any(a <= 0) for a in (dx, cosine, meridional))):
            raise ValueError('fixed_partial_v1 requires finite positive compatible spherical metrics')
        precision = np.result_type(dx.dtype, cosine.dtype)
        tolerance = max(1e-12, 8 * np.finfo(precision).eps) if precision.kind == 'f' else 1e-12
        longitude_length = dx.astype(float) / cosine.astype(float)[None, :]
        if not np.allclose(longitude_length, longitude_length[0, 0], rtol=tolerance, atol=0):
            raise ValueError('fixed_partial_v1 requires compatible dx=L*cos_lat and area=dx*dy')
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
    face_contacts = contact_depths = node_depths = None

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

    if column_geometry == 'fixed_partial_v1':
        cells = fixed_reference_nodal_cells(grid.z, grid.depth, grid.wet_mask)
        if not np.array_equal(np.asarray(wet_mask_z), cells['wet_node_mask']):
            raise ValueError('fixed partial geometry disagrees with grid wet nodes')
        wet_mask_z = jnp.asarray(cells['wet_node_mask'], dtype=dx_2d.dtype)
        # Dry widths are safe denominators, never water: every inventory and
        # face thickness multiplies by the wet mask before use.
        dz_node = jnp.asarray(np.where(cells['wet_node_mask'], cells['thickness_m'], 1.))
        dz_surface = dz_node[..., :1]
        bottom_mask = jnp.asarray(cells['bottom_node_mask'], dtype=dx_2d.dtype)
        contacts, depths = [], []
        for axis in (0,1):
            following_top = jnp.roll(jnp.asarray(cells['cell_top_m']),-1,axis=axis)
            following_bottom = jnp.roll(jnp.asarray(cells['cell_bottom_m']),-1,axis=axis)
            following_wet = jnp.roll(wet_mask_z,-1,axis=axis)
            weights, midpoints = [], []
            for offset in OFFSETS:
                top = jnp.maximum(jnp.asarray(cells['cell_top_m']),shift_vertical(following_top,offset))
                bottom = jnp.minimum(jnp.asarray(cells['cell_bottom_m']),shift_vertical(following_bottom,offset))
                weight = jnp.maximum(bottom-top,0.)*wet_mask_z*shift_vertical(following_wet,offset)
                if axis==1:
                    weight = weight.at[:,-1].set(0.)
                weights.append(weight)
                midpoints.append(jnp.where(weight>0.,.5*(top+bottom),0.))
            contacts.append(jnp.stack(weights))
            depths.append(jnp.stack(midpoints))
        face_contacts, contact_depths = tuple(contacts), tuple(depths)
        node_depths = jnp.asarray(cells['node_depth_m']).reshape(1,1,-1)

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
        face_contacts=face_contacts, contact_depths_m=contact_depths, node_depth_m=node_depths,
        column_geometry=column_geometry,
    )
