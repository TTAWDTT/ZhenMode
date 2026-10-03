"""Pure spherical grid preparation, metric construction and wet boundaries."""
from collections import deque

import numpy as np

from ocean_solver._compat import preserve_legacy_names
from ocean_solver.config.definitions import OMEGA, R_EARTH
from ocean_solver.geometry.types import GlobalOceanGrid


def _smooth_depth_once(depth):
    """One Laplacian smoothing pass on the ocean depth field.

    Each ocean cell's depth becomes the mean of itself and its 4 neighbours
    (lon-periodic, lat-bounded; land neighbours contribute depth 0). Land
    cells stay 0. This damps steep topographic gradients without moving the
    coastline (a cell that was ocean stays >= 0; a near-zero cell may later
    be re-masked as land by make_global_grid's depth>0 test). Operates on
    (nx, ny) = (lon, lat).
    """
    d = np.array(depth, dtype=np.float64)
    # 4-neighbour mean with lon wrap (axis 0), lat clamp (axis 1).
    left = np.roll(d, 1, axis=0)
    right = np.roll(d, -1, axis=0)
    up = np.empty_like(d)
    up[:, 1:] = d[:, :-1]
    up[:, 0] = d[:, 0]
    down = np.empty_like(d)
    down[:, :-1] = d[:, 1:]
    down[:, -1] = d[:, -1]
    nb_mean = 0.25 * (left + right + up + down)
    # Blend toward neighbour mean only at ocean cells; land stays 0.
    ocean = d > 0.0
    out = np.where(ocean, 0.5 * d + 0.5 * nb_mean, 0.0)
    return np.maximum(out, 0.0)


def global_grid_dims(resolution, lat_max, etopo_nlon=3600, etopo_nlat=1800,
                     etopo_res=0.1, remap="legacy"):
    """Horizontal dimensions the ETOPO reader will produce at this resolution.

    ``legacy`` mirrors the historical INTEGER-floor block division exactly:
    ``n = len(source) // step``, NOT ``round(len(source)/step)``. The two differ
    whenever step does not divide the source length (e.g. 1.3° -> step=13 ->
    3600//13 = 276, but 360/1.3 = 277), and since make_global_grid asserts the
    read matches gc.nx/gc.ny, using round() there would reject valid grids.

    ``area`` is the continuous-resolution mode.  It closes the longitude band
    exactly by using ``nx = round(360/res)`` and ``dlon_eff = 360/nx``; latitude
    likewise uses ``dlat_eff = 180/nlat_full``.  This lets --resolution accept
    0.37°, 0.85°, etc., while preserving a physically consistent periodic
    domain.  The returned nx/ny are therefore exactly the reader's output.
    """
    if remap == "legacy":
        step = int(round(resolution / etopo_res))
        if step < 1:
            raise ValueError(f"resolution {resolution} below the {etopo_res}° "
                             f"source grid")
        nx = etopo_nlon // step
        nlat_full = etopo_nlat // step
        # Lat centers sit at -90 + step*etopo_res*(k + 0.5); keep |lat| <= lat_max.
        lat_c = -90.0 + step * etopo_res * (0.5 + np.arange(nlat_full))
        ny = int(np.sum(np.abs(lat_c) <= lat_max))
        return int(nx), int(ny)
    if remap == "area":
        if resolution <= 0.0:
            raise ValueError(f"resolution must be positive (got {resolution})")
        nx = int(round(360.0 / resolution))
        nlat_full = int(round(180.0 / resolution))
        if nx < 1 or nlat_full < 1:
            raise ValueError(f"resolution {resolution} too coarse")
        dlat_eff = 180.0 / nlat_full
        lat_c = -90.0 + dlat_eff * (0.5 + np.arange(nlat_full))
        ny = int(np.sum(np.abs(lat_c) <= lat_max))
        return int(nx), int(ny)
    raise ValueError(f"unknown remap mode {remap!r}; use 'legacy' or 'area'")


def _overlap_matrix(src_edges, dst_edges, sine_weight=False):
    """Sparse-ish dense overlap matrix mapping source cells to target cells.

    For longitude the cells have equal angular width, so the overlap fraction
    is sufficient.  For latitude on a sphere, area ∝ sin(lat); the optional
    sine weighting uses that analytic primitive to build an exact conservative
    spherical-area remap.
    """
    n_src = len(src_edges) - 1
    n_dst = len(dst_edges) - 1
    W = np.zeros((n_dst, n_src), dtype=np.float64)
    for m in range(n_dst):
        a, b = dst_edges[m], dst_edges[m + 1]
        k0 = max(0, int(np.searchsorted(src_edges, a, side='right') - 1))
        k1 = min(n_src, int(np.searchsorted(src_edges, b, side='left')))
        for k in range(k0, k1):
            lo = max(a, src_edges[k])
            hi = min(b, src_edges[k + 1])
            if hi <= lo:
                continue
            if sine_weight:
                ov = np.sin(np.radians(hi)) - np.sin(np.radians(lo))
                norm = np.sin(np.radians(b)) - np.sin(np.radians(a))
            else:
                ov = hi - lo
                norm = b - a
            if norm > 0.0:
                W[m, k] = ov / norm
    return W


def _remap_etopo_area(z_full_ll, src_lon, src_lat, resolution, lat_max):
    """Conservative spherical-area remap to an arbitrary uniform target grid.

    Longitude uses plain overlap (all longitude cells at the same latitude have
    the same physical area).  Latitude uses sin-weighted overlap, matching the
    area element cos(lat)*dlat*dlon.  This replaces integer block extraction
    without changing the physical meaning of target resolution.
    """
    nlon_src = len(src_lon)
    nlat_src = len(src_lat)
    # The ETOPO reader/npz twin stores lon as the left edge of each 0.1° cell.
    src_lon_edges = np.arange(nlon_src + 1, dtype=np.float64) * 0.1
    src_lat_edges = -90.0 + np.arange(nlat_src + 1, dtype=np.float64) * 0.1

    nx = int(round(360.0 / resolution))
    nlat_full = int(round(180.0 / resolution))
    if nx < 1 or nlat_full < 1:
        raise ValueError(f"resolution {resolution} too coarse")
    dlon_eff = 360.0 / nx
    dlat_eff = 180.0 / nlat_full

    dst_lon_edges = np.arange(nx + 1, dtype=np.float64) * dlon_eff
    dst_lat_edges_full = -90.0 + np.arange(nlat_full + 1, dtype=np.float64) * dlat_eff
    lat_c_all = -90.0 + dlat_eff * (0.5 + np.arange(nlat_full))
    keep = np.abs(lat_c_all) <= lat_max
    j0 = int(np.argmax(keep))
    j1 = len(keep) - int(np.argmax(keep[::-1]))
    dst_lat_edges = dst_lat_edges_full[j0:j1 + 1]
    lat_c = lat_c_all[j0:j1]

    W_lon = _overlap_matrix(src_lon_edges, dst_lon_edges)
    W_lat = _overlap_matrix(src_lat_edges, dst_lat_edges, sine_weight=True)
    # z_full_ll @ W_lon.T gives (nlat_src, nx); W_lat @ that gives (ny, nx).
    mapped = W_lat @ (z_full_ll @ W_lon.T)
    depth = np.where(mapped < 0.0, -mapped, 0.0)
    lon_c = dlon_eff * (0.5 + np.arange(nx))
    return depth.T, lon_c, lat_c


def land_distance_from_land_mask(ocean_mask, connectivity=8):
    """Return cell distance to land for ocean cells.

    Land cells are 0 and ocean starts at 1. The zonal axis is periodic; the
    meridional axis is bounded. This shared metric supports diagnostic bands
    and optional coastal closures.
    """
    if connectivity == 4:
        neighbours = ((-1, 0), (1, 0), (0, -1), (0, 1))
    elif connectivity == 8:
        neighbours = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0), (1, 1))
    else:
        raise ValueError("connectivity must be 4 or 8")

    mask = np.asarray(ocean_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("ocean_mask must be a 2-D (nx, ny) array")
    dist = np.full(mask.shape, np.inf, dtype=np.float64)
    dist[~mask] = 0.0
    q = deque((i, j) for i in range(mask.shape[0])
              for j in range(mask.shape[1]) if not mask[i, j])
    while q:
        i, j = q.popleft()
        for di, dj in neighbours:
            ni, nj = (i + di) % mask.shape[0], j + dj
            if (0 <= nj < mask.shape[1] and mask[ni, nj]
                    and dist[i, j] + 1.0 < dist[ni, nj]):
                dist[ni, nj] = dist[i, j] + 1.0
                q.append((ni, nj))
    return dist


def build_global_grid(grid_config, depth, lon, lat, smooth_passes=0,
                     min_depth=None, remap="legacy"):
    """Build metrics and masks from already prepared bathymetry arrays.

    Builds a true global grid with:
      - periodic longitude (axis 0 wraps),
      - spherical metric (dx varies with latitude),
      - a real wet_mask (1=ocean, 0=land) for the FD solver's no-flux land BC,
      - full 2D Coriolis f = 2*Omega*sin(lat).
    Polar regions above lat_max are excluded (polar cap); the FD solver
    applies a polar-cap filter on the poleward-most row to handle the
    cos(lat)->0 metric singularity.

    smooth_passes: number of Laplacian smoothing passes applied to the ocean
      depth field (land stays 0). Smooths steep topographic gradients
      (continental slopes, trenches) that at 1° resolution drive an
      under-resolved topographic PGF which destabilizes the FD solver.
      Standard OGCM practice (MOM6 applies bathymetry filtering by default).
      0 = raw ETOPO (no smoothing). Each pass replaces an ocean cell's depth
      with the mean of itself + its valid (ocean or land=0) 4-neighbours,
      lon-periodic, lat-bounded. Land/sea mask is re-derived AFTER smoothing
      so a cell that smooths to ~0 becomes land (prevents flooded coast).
    min_depth: ocean points shallower than this become land (depth=0).
      Default = |z[1]| (top interior level thickness), so every wet column
      has at least 2 wet layers (surface + one interior). Ultra-shallow
      coastal points (<5m at 1°) carry unreliable WOA T and produce extreme
      horizontal gradients that destabilize the FD solver. Standard OGCM
      practice (a "minimum depth" / partial-cell floor). 0 = no floor.
    """
    gc = grid_config
    nx, ny = depth.shape
    assert nx == gc.nx, f"lon dim {nx} != config nx {gc.nx}"
    assert ny == gc.ny, f"lat dim {ny} != config ny {gc.ny}"

    # ── Bathymetry smoothing (option A) ──
    for _ in range(int(smooth_passes)):
        depth = _smooth_depth_once(depth)

    # ── Minimum-depth floor (drop ultra-shallow points) ──
    if min_depth is None:
        z = np.array(gc.z_levels, dtype=np.float64)
        min_depth = float(abs(z[1])) if len(z) > 1 else 5.0   # ~5m
    depth = np.where(depth > 0.0, np.where(depth < min_depth, 0.0, depth), 0.0)

    # ── Spherical metric ──
    # Continuous area remap closes 360° with nx cells, so dlon_eff may differ
    # from the requested value by the usual rounding residual. Legacy keeps the
    # requested spacing exactly for bit-for-bit backwards compatibility.
    dlon = 360.0 / nx if remap == "area" else gc.dlon
    dlat = (float(np.median(np.diff(lat))) if ny > 1 and remap == "area"
            else gc.dlat)
    cos_lat = np.cos(np.radians(lat))            # (ny,)
    dx_2d = np.broadcast_to(
        R_EARTH * np.radians(dlon) * cos_lat, (nx, ny)
    ).copy()                                      # (nx, ny) varies with lat
    dy = R_EARTH * np.radians(dlat)

    # ── Coriolis: f = 2*Omega*sin(lat), full 2D field ──
    lon_2d, lat_2d = np.meshgrid(lon, lat, indexing='ij')   # (nx, ny)
    f = 2.0 * OMEGA * np.sin(np.radians(lat_2d))

    # ── Vertical grid ──
    z = np.array(gc.z_levels, dtype=np.float64)
    dz = np.abs(np.diff(z))

    # ── Masks ──
    ocean_mask = depth > 0.0
    land_mask = ~ocean_mask
    wet_mask = ocean_mask.astype(np.float64)     # 1.0 ocean, 0.0 land

    # Vertical wet mask: layer k is wet iff |z[k]| <= depth (above seafloor)
    # and the column is ocean. Layers below the seafloor are dry ("ghost
    # water" excluded from pressure integration). z is negative downward
    # so |z[k]| is the depth of level k.
    abs_z = np.abs(z)                                   # (nz,) depth of each level
    wet_mask_3d = (
        (abs_z[None, None, :] <= depth[:, :, None])    # level above seafloor
        & ocean_mask[:, :, None]                        # column is ocean
    ).astype(np.float64)                                # (nx, ny, nz)

    return GlobalOceanGrid(
        lon=lon, lat=lat,
        dx_2d=dx_2d, dy=float(dy), cos_lat=cos_lat,
        f=f,
        z=z, dz=dz, nz=gc.nz,
        depth=depth, wet_mask=wet_mask,
        ocean_mask=ocean_mask, land_mask=land_mask,
        wet_mask_3d=wet_mask_3d,
        nx=nx, ny=ny,
    )


preserve_legacy_names(globals(), 'grid')
