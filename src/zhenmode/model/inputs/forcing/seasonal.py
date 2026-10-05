"""Independent NumPy and JAX controls for 360-day seasonal interpolation."""

from __future__ import annotations

from zhenmode.model.inputs.forcing.reanalysis import real_wind_forcing
from zhenmode.model.solver.numerics.backend import jnp, np


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
    """Blend zonal and meridional stress on the repeating 360-day calendar."""
    # Each component uses the host field routine without changing array dtype.
    return tuple(interp_monthly_field([month[c] for month in wind_months], day, blend_days)
                 for c in range(2))


def interp_seasonal_wind_jit(wind_stack, day, blend_days=5.0):
    """Traceable wind interpolation; stack shape is (12, 2, nx, ny)."""
    blended = interp_monthly_field_jit(wind_stack, day, blend_days)
    return blended[0], blended[1]


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
    # Zero selects the current month, including month/year boundaries. Use a
    # safe denominator even when width is traced; where alone cannot undo NaN.
    denominator = jnp.where(blend_days > 0.0, blend_days, 1.0)
    w_after = jnp.clip((mpos - (month_len - half)) / denominator, 0.0, 1.0)
    w_before = jnp.clip((half - mpos) / denominator, 0.0, 1.0)
    cur = field_stack[mi]
    nxt = field_stack[nxt_i]
    prev = field_stack[prev_i]
    blended = (1.0 - w_after - w_before) * cur + w_after * nxt + w_before * prev
    return jnp.where(blend_days > 0.0, blended, cur)
