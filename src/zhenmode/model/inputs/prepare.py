"""Resolve selected input files and construct grid, initial fields, and masks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from zhenmode.model.config import DEFAULT_CONFIG, GlobalGridConfig, PhysicsConfig
from zhenmode.model.diagnostics.mixed_layer import mixed_layer_depth
from zhenmode.model.runtime.cli import (
    DT_BT_DEFAULT,
    NU_BI_DEFAULT,
    NU_H_DEFAULT,
    NY_DEFAULT,
    scaled_physics_for_resolution,
)
from zhenmode.model.solver.geometry.grid import (
    GlobalOceanGrid,
    global_grid_dims,
    land_distance_from_land_mask,
)
from zhenmode.model.solver.numerics.backend import np


def _input_files(args, *, seasonal=None, air=None, default_config=DEFAULT_CONFIG):
    """Selected loader paths, including WOA's documented npz precedence."""
    import zhenmode.model.inputs.bathymetry as grid_module
    import zhenmode.model.inputs.forcing.reanalysis as reanalysis
    import zhenmode.model.inputs.initial_conditions as woa_data

    bathy = Path(default_config.bathymetry_file)
    if (not bathy.is_file() or grid_module.Dataset is None) and Path(str(bathy) + ".npz").is_file():
        bathy = Path(str(bathy) + ".npz")
    paths = {"bathymetry": bathy}
    if args.init_from:
        paths["initial_fields"] = Path(args.init_from)
    else:
        for name, filename in woa_data.WOA_FILES.items():
            path = Path(filename)
            paths[name] = Path(filename + ".npz") if Path(filename + ".npz").is_file() else path
    seasonal = args.seasonal_wind if seasonal is None else seasonal
    months = range(12) if seasonal else [int(args.month[5:7]) - 1]
    year = args.wind_year if seasonal else int(args.month[:4])
    for month in months:
        index = (year - 1948) * 12 + month
        paths[f"wind_{index}"] = Path(reanalysis.WIND_CACHE_DIR) / f"monthly_mean_{index}.npz"
    if not args.no_bulk_flux and args.lambda_bulk * args.bulk_lambda_mult > 0:
        if air is None:
            air = (
                "monthly"
                if args.real_air_temp_monthly
                else "annual"
                if args.real_air_temp
                else "zonal"
            )
        if air in {"monthly", "annual"}:
            paths["air"] = Path(reanalysis.AIR_CACHE_DIR) / f"air_2m_{air}_{args.wind_year:04d}.npz"
    return paths


def _lat_band_mask(grid, band):
    """Build a 2D mixed-layer mask from a inclusive latitude band."""
    lat_min, lat_max = band
    return ((grid.lat[None, :] >= lat_min) & (grid.lat[None, :] <= lat_max)).astype(float)


@dataclass
class GridInputs:
    gcfg: GlobalGridConfig
    grid: GlobalOceanGrid
    ocean: np.ndarray
    dx_eq: float
    physics: PhysicsConfig
    dt_bt: float
    T_init: np.ndarray
    S_init: np.ndarray
    T_init_max: float
    Q_heat: np.ndarray
    stratification_mld: np.ndarray | None
    coastal_bulk_mask: np.ndarray | None
    coastal_kappa_v_mask: np.ndarray | None
    coastal_kappa_h_mask: np.ndarray | None
    coastal_restore_mask: np.ndarray | None


def load_grid_inputs(args, ap, services):
    bathy = services.default_config.bathymetry_file
    if args.resolution is not None:
        res = float(args.resolution)
        if res <= 0.0:
            ap.error(f"--resolution must be positive (got {res})")
        if args.resolution_remap == "legacy":
            step = res / 0.1
            if abs(step - round(step)) > 1e-09:
                ap.error(
                    f"--resolution must be a multiple of the 0.1° ETOPO source grid when --resolution-remap=legacy (got {res}; use --resolution-remap=area)"
                )
        resolve_dimensions = services.resolve_grid_dimensions or global_grid_dims
        nx, ny = resolve_dimensions(res, args.lat_max, remap=args.resolution_remap)
        if nx < 4 or ny < 4:
            ap.error(
                f"--resolution {res} at lat_max={args.lat_max} gives a {nx}x{ny} grid; too coarse"
            )
        gcfg_kwargs = {"lat_max": args.lat_max, "ny": ny, "nx": nx, "resolution": res}
        if args.ny is not None:
            print(f"  NOTE: --ny {args.ny} ignored (--resolution {res} derives ny={ny})")
    else:
        ny = NY_DEFAULT if args.ny is None else args.ny
        gcfg_kwargs = {"lat_max": args.lat_max, "ny": ny}
    if args.resolution is not None:
        dt_bt, nu_h, nu_bi = scaled_physics_for_resolution(
            args.resolution, args.dt_bt, args.nu_h, args.nu_bi
        )
        overrides = []
        if args.dt_bt is None:
            overrides.append(f"dt_bt={dt_bt:.0f}s")
        if args.nu_h is None:
            overrides.append(f"nu_h={nu_h:.3g}")
        if args.nu_bi is None:
            overrides.append(f"nu_bi={nu_bi:.3g}")
        if overrides:
            print(
                f"  resolution {args.resolution:g}° auto-scales {', '.join(overrides)} (1° CFL values x dx^p)"
            )
    else:
        dt_bt = DT_BT_DEFAULT if args.dt_bt is None else args.dt_bt
        nu_h = NU_H_DEFAULT if args.nu_h is None else args.nu_h
        nu_bi = NU_BI_DEFAULT if args.nu_bi is None else args.nu_bi
    if args.z_levels:
        zl = tuple((float(v) for v in args.z_levels.split(",")))
        assert (
            len(zl) >= 3 and zl[0] == 0.0 and all((zl[i] > zl[i + 1] for i in range(len(zl) - 1)))
        ), "bad --z-levels"
        gcfg_kwargs["z_levels"] = zl
        gcfg_kwargs["nz"] = len(zl)
        print(f"  vertical override: {len(zl)} levels, z={zl}")
    gcfg = replace(GlobalGridConfig(), **gcfg_kwargs)
    print(
        f"Building global FD grid (lat_max={args.lat_max}, resolution={gcfg.resolution}°, ny={gcfg.ny}, smooth={args.smooth_passes}, min_depth={args.min_depth})..."
    )
    grid = services.make_global_grid(
        gcfg,
        bathy,
        smooth_passes=args.smooth_passes,
        min_depth=args.min_depth,
        remap=args.resolution_remap,
    )
    ocean = np.asarray(grid.ocean_mask, dtype=bool)
    dx_eq = float(grid.dx_2d[0, grid.ny // 2])
    print(
        f"  grid {grid.nx}x{grid.ny}x{grid.nz}, ocean {float(grid.wet_mask.mean()):.1%}, dx_eq={dx_eq:.0f}m, lat[{grid.lat[0]:.1f},{grid.lat[-1]:.1f}]"
    )
    physics = replace(
        PhysicsConfig(),
        nu_h=nu_h,
        nu_bi=nu_bi,
        kappa_bi=nu_bi,
        kappa_gm=args.kappa_gm,
        kappa_redi=args.kappa_redi,
        kappa_v=args.kappa_v if args.kappa_v is not None else 1e-05,
        kappa_conv=args.kappa_conv,
        gm_slope_max=args.gm_slope_max,
    )
    Q_heat = services.heat_flux_meridional(grid, Q0=50.0)
    if args.no_meridional_heat_flux:
        Q_heat = np.zeros_like(Q_heat)
    if args.init_from:
        print(f"Loading precomputed init fields from {args.init_from}...")
        zf = np.load(args.init_from)
        T_init = np.array(zf["T_init"], dtype=np.float64)
        S_init = np.array(zf["S_init"], dtype=np.float64)
    else:
        print("Loading WOA2023 climatology for initial T/S...")
        T_init, S_init = services.get_initial_fields(grid)
    T_init = np.array(T_init)
    S_init = np.array(S_init)
    T_init_max = float(np.max(T_init))
    nan_init = int(np.isnan(T_init).sum() + np.isnan(S_init).sum())
    if nan_init > 0:
        print(f"  WARNING: {nan_init} NaN in initial fields (should be 0 after fill)")
    print(f"  T_init range=[{T_init.min():.2f}, {T_init_max:.2f}] C")
    stratification_mld = None
    if args.mixed_layer_mode == "stratification":
        if args.mixed_layer_depth is None:
            raise SystemExit(
                "--mixed-layer-mode stratification requires --mixed-layer-depth as an upper/backstop depth"
            )
        raw_mld = mixed_layer_depth(
            T_init,
            S_init,
            np.asarray(grid.z),
            ocean=ocean,
            density_delta=float(args.mld_density_delta),
        )
        mld_lo = max(0.0, float(args.mixed_layer_depth_min))
        mld_hi = float(args.mixed_layer_depth_max)
        if not np.isfinite(mld_lo) or not np.isfinite(mld_hi) or mld_hi <= mld_lo:
            raise SystemExit("--mixed-layer-depth-max must exceed --mixed-layer-depth-min")
        fallback = float(np.clip(np.nanmedian(raw_mld[ocean]), mld_lo, mld_hi))
        stratification_mld = np.where(
            ocean & np.isfinite(raw_mld), np.clip(raw_mld, mld_lo, mld_hi), fallback
        )
        print(
            f"  stratification MLD: threshold={args.mld_density_delta:g} kg/m^3, clip=[{mld_lo:g},{mld_hi:g}]m, mean={float(np.mean(stratification_mld[ocean])):.1f}m, p10/p50/p90={np.nanpercentile(stratification_mld[ocean], [10, 50, 90]).round(1)}"
        )
    dist = None
    def coastal_distance():
        nonlocal dist
        if dist is None:
            dist = land_distance_from_land_mask(ocean, connectivity=8)
        return dist

    coastal_bulk_mask = None
    if args.coastal_bulk_lambda > 0.0 and args.coastal_bulk_cells > 0:
        dist = coastal_distance()
        coastal_bulk_mask = ocean & (dist <= float(args.coastal_bulk_cells))
        print(
            f"  coastal bulk flux: lambda={args.coastal_bulk_lambda:g} W/m^2/K, cells<={args.coastal_bulk_cells}, n={int(coastal_bulk_mask.sum())}"
        )
    coastal_kappa_v_mask = None
    if args.coastal_kappa_v > 0.0 and args.coastal_kappa_v_cells > 0:
        dist = coastal_distance()
        coastal_kappa_v_mask = ocean & (dist <= float(args.coastal_kappa_v_cells))
        print(
            f"  coastal kappa_v: +{args.coastal_kappa_v:g} m^2/s, cells<={args.coastal_kappa_v_cells}, n={int(coastal_kappa_v_mask.sum())}"
        )
    coastal_kappa_h_mask = None
    if args.coastal_kappa_h > 0.0 and args.coastal_kappa_h_cells > 0:
        dist = coastal_distance()
        coastal_kappa_h_mask = ocean & (dist <= float(args.coastal_kappa_h_cells))
        print(
            f"  coastal kappa_h: +{args.coastal_kappa_h:g} m^2/s, cells<={args.coastal_kappa_h_cells}, n={int(coastal_kappa_h_mask.sum())}"
        )
    coastal_restore_mask = None
    if args.coastal_restore_days > 0.0 and args.coastal_restore_cells > 0:
        dist = coastal_distance()
        band = ocean & (dist <= float(args.coastal_restore_cells))
        coastal_restore_mask = np.zeros(ocean.shape, dtype=np.float64)
        taper = args.coastal_restore_taper
        if taper == "none":
            coastal_restore_mask[band] = 1.0
        else:
            dmax = max(float(args.coastal_restore_cells), 1.0)
            if taper == "linear":
                weight = np.clip(1.0 - dist / dmax, 0.0, 1.0)
            else:
                weight = 0.5 * (1.0 + np.cos(np.pi * np.clip(dist / dmax, 0.0, 1.0)))
            coastal_restore_mask = ocean * weight
        print(
            f"  coastal T restore: tau={args.coastal_restore_days:g}d, cells<={args.coastal_restore_cells}, taper={taper}, n={int(np.count_nonzero(coastal_restore_mask))}"
        )
    if args.global_sst_restore_days > 0.0:
        if coastal_restore_mask is not None:
            raise SystemExit(
                "--global-sst-restore-days cannot be combined with --coastal-restore-days"
            )
        coastal_restore_mask = ocean.astype(float)
        print(f"  global SST restore: tau={args.global_sst_restore_days:g}d over all wet cells")
    return GridInputs(
        gcfg=gcfg,
        grid=grid,
        ocean=ocean,
        dx_eq=dx_eq,
        physics=physics,
        dt_bt=dt_bt,
        T_init=T_init,
        S_init=S_init,
        T_init_max=T_init_max,
        Q_heat=Q_heat,
        stratification_mld=stratification_mld,
        coastal_bulk_mask=coastal_bulk_mask,
        coastal_kappa_v_mask=coastal_kappa_v_mask,
        coastal_kappa_h_mask=coastal_kappa_h_mask,
        coastal_restore_mask=coastal_restore_mask,
    )
