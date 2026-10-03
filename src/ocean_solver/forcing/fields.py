"""
Surface Forcing Module — heat flux and meridional taper helpers.

The global run takes its wind from NCEP/NCAR R1 reanalysis
(wind_reanalysis.py); this module supplies the remaining surface forcing:

  - Q_heat          : idealized meridional heat-flux pattern, ocean-mean
                      centred (heat_flux_meridional)
  - T_atm           : zonally-uniform atmospheric target for the bulk
                      air-sea flux (air_temp_profile)
  - ocean_zonal_mean: the shared ocean-only zonal mean both targets are
                      built from (land fill never enters)
  - meridional taper: raised-cosine edge ramp (taper_weight_1d, taper_2d_y)

All fields are (nx, ny) shaped, matching the solver axis convention
(axis 0 = zonal/lon, axis 1 = meridional/lat).
"""

import numpy as np

# Edge-taper width [grid cells] for the idealized forcing fields
# (meridional heat flux, wind stress). Ramping them to zero over the
# outermost rows stops them fighting the polar-edge sponge / cap band.
FORCING_TAPER_CELLS = 8


def taper_weight_1d(ny, taper_cells):
    """Raised-cosine y-edge taper weight (multiplicative mask).

    Returns a ``(ny,)`` weight that is 1 in the interior and ramps smoothly
    to 0 over ``taper_cells`` cells at each meridional edge. Used to zero
    idealized forcing in the polar-edge sponge / cap band (see _taper_y).
    """
    taper_cells = int(taper_cells)
    weight = np.ones(int(ny), dtype=np.float64)
    if taper_cells <= 0:
        return weight
    taper_cells = min(taper_cells, int(ny) // 2)
    taper = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, taper_cells)))
    weight[:taper_cells] = taper
    weight[-taper_cells:] = taper[::-1]
    return weight


def _taper_y(profile, ny, taper_cells):
    """Ramp a 1D y-profile smoothly to zero at both meridional edges.

    Legacy helper. On a y-periodic grid a non-zero edge value puts a step
    discontinuity at the FFT seam and pumps grid-scale energy there. The
    global grid has CLOSED no-flux N/S walls instead, so there is no seam
    to close. The taper survives because the polar edge rows are the
    sponge / polar-cap band, and ramping the idealized forcing to zero
    there stops it fighting them. It is a raised cosine over
    ``taper_cells`` cells at each edge.

    Args:
        profile: (ny,) 1D meridional profile before tapering.
        ny: number of meridional grid cells.
        taper_cells: number of edge cells over which the profile is
            smoothly ramped to zero at each boundary.

    Returns:
        (ny,) tapered profile.
    """
    return np.asarray(profile, dtype=np.float64) * taper_weight_1d(ny, taper_cells)


def taper_2d_y(field, ny, taper_cells):
    """Apply the meridional y-edge taper to a 2D ``(nx, ny)`` forcing field.

    Broadcasts the 1D taper weight along axis 1 so the whole field reaches
    zero at both y-edges (see _taper_y for why the global grid still uses it).
    """
    field = np.asarray(field, dtype=np.float64)
    return field * taper_weight_1d(ny, taper_cells)[None, :]

def heat_flux_meridional(grid, Q0=50.0):
    """Meridional heat flux gradient.

    Q(y) = -Q0 * (2*y' - 1)

    where y' is normalized meridional coordinate (0=south, 1=north).
    This gives:
      - Warming (Q > 0) in the south
      - Cooling (Q < 0) in the north
      - Zero at the center

    Args:
        grid: GlobalOceanGrid (uses nx, ny, wet_mask, dx_2d, dy)
        Q0: peak heat flux magnitude [W/m^2], default 50

    Returns:
        Q_heat: (nx, ny) heat flux field [W/m^2]
    """
    ny = grid.ny
    taper_cells = FORCING_TAPER_CELLS
    y_frac = np.arange(ny, dtype=np.float64) / (ny - 1)

    q_profile = -Q0 * (2.0 * y_frac - 1.0)  # (ny,)
    q_profile = _taper_y(q_profile, ny, taper_cells)
    Q = np.broadcast_to(q_profile[None, :], (grid.nx, ny)).astype(np.float64).copy()
    # The surface heat term is land-masked, so a pattern whose GLOBAL mean is
    # zero still carries a large OCEAN mean. The 1-deg bathymetry is
    # hemispherically asymmetric in the +/-60 band: the Southern Ocean
    # (oceanfrac ~1.0 at -50) collects the full southern warm lobe while the
    # northern cool lobe falls largely on land (oceanfrac 0.39-0.53 at
    # +40..+50). The raw pattern measured +4.68 W/m^2 over ocean = +1511 TW =
    # +47.7 ZJ/yr of spurious heat -- a ~48-yr time constant that would
    # dominate any real equilibration. Recentre on the ocean-area-weighted
    # mean so the flux applied actually conserves heat over the region it is
    # applied to.
    wm = np.asarray(grid.wet_mask, dtype=np.float64)
    w = wm * (np.asarray(grid.dx_2d, dtype=np.float64) * float(grid.dy))
    Q = (Q - float((Q * w).sum() / w.sum())) * wm
    return Q


# ── Bulk air-sea heat flux (Haney/Barnier formulation) ───────────────
# The fixed Q_heat_meridional pattern injects up to ±50 W/m^2 with NO
# dependence on the surface temperature itself: a column that warms does
# not lose more heat, so there is no negative feedback to arrest a
# runaway surface hotspot. Under the linear EOS warm surface water is
# always lighter (stably stratified), so convective adjustment never
# fires, and kappa_v=1e-5 removes surface heat ~30x slower than Q_heat
# injects it. The only thing holding the SST down was the separate Haney
# restore term — i.e. the climatology skill was restore-manufactured.
#
# The physically correct fix is a bulk air-sea heat exchange with genuine
# SST feedback, as in Haney (1971) / Barnier (1995) bulk formulations:
#
#     Q_net = Q_clim(y) + lambda_bulk * (T_atm - T_sst)
#
# The lambda_bulk*(T_atm - T_sst) term is computed inside the tracer
# tendency from the live surface T (it cannot be a static field). It is
# net-balanced in equilibrium (T_sst -> T_atm - Q_clim/lambda) and gives
# real negative feedback: a warmed column loses heat faster. This lets
# the model hold a stable SST *without* the restore crutch.
#
# To keep the A1/A2 climatology comparison non-circular, T_atm is built
# as a ZONALLY-UNIFORM meridional profile — only the large-scale
# meridional gradient is prescribed (the forced part). Any zonal SST
# structure the model produces is genuinely predicted by its own
# advection/mixing, so A2 (a 2D pattern test) retains discriminative
# power rather than measuring how well SST tracks a 2D clamp.

BULK_LAMBDA_DEFAULT = 40.0  # W/m^2/K — Haney/Barnier bulk transfer coef
# Over the 5 m surface layer this is a ~30-day e-folding timescale
# (lambda / (rho0*cp*dz) = 40/(1025*3985*5) = 1.96e-6 1/s ~ 5.9 d...
# actually faster; see _compute_tracer_tendency heat_factor). It is
# strong enough to kill an 80-day hotspot but weak enough that
# advection/mixing can move SST off the target — a real equilibrium,
# not a clamp.


def ocean_zonal_mean(grid, field):
    """Ocean-only zonal mean of a 2D ``(nx, ny)`` field: one value per lat row.

    Averages over wet columns only, so land and below-seafloor fill values
    never enter, and backfills an all-land row from the nearest ocean-bearing
    rows by linear interpolation, so the profile is defined at every
    latitude. Both properties are load-bearing for the non-circularity of the
    A1/A2 skill test (see air_temp_profile).
    """
    ny = grid.ny
    wm = np.asarray(grid.wet_mask, dtype=np.float64)   # (nx, ny)
    f = np.asarray(field, dtype=np.float64)
    profile = np.full(ny, np.nan)
    for j in range(ny):
        wet_j = wm[:, j] > 0.5
        if wet_j.any():
            profile[j] = f[wet_j, j].mean()
    # backfill any all-land row from its nearest ocean-bearing row
    if np.isnan(profile).any():
        good = np.where(~np.isnan(profile))[0]
        profile = np.interp(np.arange(ny), good, profile[good])
    return profile


def air_temp_profile(grid, sst_clim):
    """Zonally-uniform meridional atmospheric target temperature.

    Builds the bulk-flux atmospheric equilibrium temperature T_atm as the
    OCEAN-ONLY zonal mean of a climatological SST field (ocean_zonal_mean).
    Zonally uniform by construction, so only the meridional gradient is
    prescribed and any zonal SST structure is left for the model to predict
    (keeps A1/A2 non-circular).

    Two measured biases this construction avoids (both on gpu365_glap):
      1. A plain zonal mean of T_init includes the land / below-seafloor fill
         values (init_state holds +15 C sentinels; the WOA interp holds its
         own land fill), which dragged the equatorial T_atm down to 24.1 C vs
         the true ocean-only 27.4 C. The model SST then equilibrated exactly
         onto the polluted target (model 24.09 vs T_atm 24.03) -- the
         measured -3..-5 K tropical cold bias.
      2. Tapering the edge rows toward the domain mean pulled |lat|~59.5 to
         17.1 C vs the true -0.8 C, injecting +0.89 K/d of spurious polar
         warming -- the measured +2..+4 K polar warm bias. The global grid's
         y axis is a real closed boundary, not a periodic seam, so the bulk
         flux is already continuous there and needs no taper.

    Args:
        grid: GlobalOceanGrid (uses nx, ny, wet_mask)
        sst_clim: (nx, ny) climatological surface T [degC] (e.g. WOA SST).

    Returns:
        T_atm: (nx, ny) atmospheric target temperature [degC].
    """
    profile = ocean_zonal_mean(grid, sst_clim)
    return np.broadcast_to(profile[None, :], (grid.nx, grid.ny)).copy()
