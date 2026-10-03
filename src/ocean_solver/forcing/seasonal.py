"""Independent NumPy and JAX controls for 360-day seasonal interpolation."""

from __future__ import annotations

from ocean_solver.forcing.wind import real_wind_forcing
from ocean_solver.numerics.backend import jnp, np


def marine_smooth_2d(field, ocean_mask, passes=0):
    """Smooth a 2-D forcing field using only wet-cell neighbours.

    This is a simple marine-air correction: it reduces land-contaminated
    coastal extrema without interpolating temperature through land.
    """
    if passes <= 0:
        return np.asarray(field, dtype=np.float64)
    mask = np.asarray(ocean_mask, dtype=bool)
    y = np.asarray(field, dtype=np.float64).copy()
    for _ in range(int(passes)):
        s = np.zeros_like(y, dtype=np.float64)
        c = np.zeros_like(y, dtype=np.int64)
        for src_mask, src in (
            (np.roll(mask, -1, axis=0), np.roll(y, -1, axis=0)),
            (np.roll(mask, 1, axis=0), np.roll(y, 1, axis=0)),
            (np.roll(mask, -1, axis=1), np.roll(y, -1, axis=1)),
            (np.roll(mask, 1, axis=1), np.roll(y, 1, axis=1)),
        ):
            valid = mask & src_mask
            s[valid] += src[valid]
            c[valid] += 1
        avg = np.zeros_like(y)
        good = mask & (c > 0)
        avg[good] = s[good] / c[good]
        y = np.where(good, 0.5 * y + 0.5 * avg, y)
    return y


def build_seasonal_wind_global(grid, year=2023):
    """12 monthly NCEP wind-stress snapshots interpolated to the global grid.

    real_wind_forcing is grid-agnostic (bilinear interp on grid.lat/lon);
    the global grid's ±lat_max range is within NCEP lat coverage. The y-edge
    taper is applied per snapshot (taper_2d_y) — compatible with the no-flux
    wall (smooths the boundary anomaly, doesn't conflict with v=0 at the wall).
    """
    months = []
    for m in range(1, 13):
        month_idx = (year - 1948) * 12 + (m - 1)
        tau_x, tau_y = real_wind_forcing(month_idx=month_idx, grid=grid)
        months.append((tau_x, tau_y))
    return months


def interp_seasonal_wind(wind_months, day, blend_days=5.0):
    """Linearly blend monthly wind snapshots near 30-day month boundaries.

    Removes the artificial
    step discontinuity at month transitions that excited boundary instabilities.
    """
    month_len = 30.0
    mpos = day % month_len
    mi = int(day // month_len) % 12
    half = blend_days / 2.0
    if blend_days <= 0.0 or (mpos >= half and mpos <= month_len - half):
        return wind_months[mi]
    if mpos < half:
        w = (half + mpos) / blend_days
        prev = wind_months[(mi - 1) % 12]
        cur = wind_months[mi]
        return ((1.0 - w) * prev[0] + w * cur[0], (1.0 - w) * prev[1] + w * cur[1])
    w = (mpos - (month_len - half)) / blend_days
    cur = wind_months[mi]
    nxt = wind_months[(mi + 1) % 12]
    return ((1.0 - w) * cur[0] + w * nxt[0], (1.0 - w) * cur[1] + w * nxt[1])


def interp_seasonal_wind_jit(wind_stack, day, blend_days=5.0):
    """JIT-traceable version of interp_seasonal_wind.

    wind_stack: (12, nx, ny) constant on device; day is a traced scalar.
    Reproduces the same 30-day month grid + blend_days tanh-free linear
    crossfade as interp_seasonal_wind, without Python branching on `day`
    (which would force a retrace per step and a fresh H2D copy of the
    blended field).
    """
    month_len = 30.0
    mpos = day % month_len
    mi = jnp.floor(day / month_len).astype(jnp.int32) % 12
    prev_i = (mi - 1) % 12
    nxt_i = (mi + 1) % 12
    half = blend_days / 2.0
    # Blend windows: w=0 -> pure current month; rises to 1 at the edges.
    w_after = jnp.clip((mpos - (month_len - half)) / blend_days, 0.0, 1.0)
    w_before = jnp.clip((half - mpos) / blend_days, 0.0, 1.0)
    cur = wind_stack[mi]
    nxt = wind_stack[nxt_i]
    prev = wind_stack[prev_i]
    # Blend: pure current month in the interior, linear crossfade to the
    # previous month across the leading edge and to the next month across
    # the trailing edge (weights are 0 outside the blend windows).
    tx = (1.0 - w_after - w_before) * cur[0] + w_after * nxt[0] + w_before * prev[0]
    ty = (1.0 - w_after - w_before) * cur[1] + w_after * nxt[1] + w_before * prev[1]
    return tx, ty


def interp_monthly_field(fields, day, blend_days=5.0):
    """Blend a single field through the same 30-day seasonal calendar as wind."""
    month_len = 30.0
    mpos = day % month_len
    mi = int(day // month_len) % 12
    half = blend_days / 2.0
    if blend_days <= 0.0 or (mpos >= half and mpos <= month_len - half):
        return fields[mi]
    if mpos < half:
        w = (half + mpos) / blend_days
        prev = fields[(mi - 1) % 12]
        cur = fields[mi]
        return (1.0 - w) * prev + w * cur
    w = (mpos - (month_len - half)) / blend_days
    cur = fields[mi]
    nxt = fields[(mi + 1) % 12]
    return (1.0 - w) * cur + w * nxt


def interp_monthly_field_jit(field_stack, day, blend_days=5.0):
    """JIT-traceable single-field version of the seasonal wind blending."""
    month_len = 30.0
    mpos = day % month_len
    mi = jnp.floor(day / month_len).astype(jnp.int32) % 12
    prev_i = (mi - 1) % 12
    nxt_i = (mi + 1) % 12
    half = blend_days / 2.0
    w_after = jnp.clip((mpos - (month_len - half)) / blend_days, 0.0, 1.0)
    w_before = jnp.clip((half - mpos) / blend_days, 0.0, 1.0)
    cur = field_stack[mi]
    nxt = field_stack[nxt_i]
    prev = field_stack[prev_i]
    return (1.0 - w_after - w_before) * cur + w_after * nxt + w_before * prev
