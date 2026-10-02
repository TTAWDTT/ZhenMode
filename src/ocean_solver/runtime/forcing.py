"""Load effective forcing and provenance, then bind monthly step dispatch."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from forcing import ocean_zonal_mean
from ocean_solver.fd.backend import jax, jnp, np
from ocean_solver.runtime.seasonal import (
    interp_monthly_field,
    interp_monthly_field_jit,
    interp_seasonal_wind,
    interp_seasonal_wind_jit,
    marine_smooth_2d,
)
from restart_contract import file_sha256, fingerprint


@dataclass
class ForcingBundle:
    Q_heat: np.ndarray
    seasonal: bool
    wind_months: list[tuple[np.ndarray, np.ndarray]] | None
    wind_src: str
    lambda_bulk: float
    T_atm: np.ndarray | None
    T_atm_months: np.ndarray | None
    T_atm_source: str
    S_ref_surf: np.ndarray | None
    forcing_baked: tuple[np.ndarray, np.ndarray, np.ndarray]
    forcing_provenance: dict
    source_identity: dict
    execution_identity: dict

    def bind_step(self, args, step, step_dyn=None):
        _fdtype = jnp.float32 if args.dtype == "float32" else jnp.float64
        if self.seasonal:
            Q_heat_2d = jnp.array(self.Q_heat, dtype=_fdtype)
            if self.T_atm_months is not None:
                air_stack = jnp.array(np.asarray(self.T_atm_months), dtype=_fdtype)
            if args.wind_jit:
                wind_stack = jnp.array(
                    np.stack([np.stack(m) for m in self.wind_months]), dtype=_fdtype
                )

                def do_step(state, month_day):
                    tx, ty = interp_seasonal_wind_jit(
                        wind_stack, month_day, blend_days=args.wind_blend_days
                    )
                    if self.T_atm_months is not None:
                        ta = interp_monthly_field_jit(
                            air_stack, month_day, blend_days=args.wind_blend_days
                        )
                        return step_dyn(state, tx, ty, Q_heat_2d, T_atm_3d=ta[:, :, None])
                    return step_dyn(state, tx, ty, Q_heat_2d)
            else:

                def do_step(state, month_day):
                    tx, ty = interp_seasonal_wind(
                        self.wind_months, month_day, blend_days=args.wind_blend_days
                    )
                    if self.T_atm_months is not None:
                        ta = np.asarray(
                            interp_monthly_field(
                                self.T_atm_months, month_day, blend_days=args.wind_blend_days
                            ),
                            dtype=np.float64,
                        )
                        return step_dyn(
                            state,
                            jnp.array(tx, dtype=_fdtype),
                            jnp.array(ty, dtype=_fdtype),
                            Q_heat_2d,
                            T_atm_3d=jnp.array(ta, dtype=_fdtype)[:, :, None],
                        )
                    return step_dyn(
                        state, jnp.array(tx, dtype=_fdtype), jnp.array(ty, dtype=_fdtype), Q_heat_2d
                    )
        else:

            def do_step(state, month_day):
                return step(state)

        return do_step


def load_forcing(args, inputs, services):
    grid = inputs.grid
    T_init = inputs.T_init
    S_init = inputs.S_init
    Q_heat = inputs.Q_heat
    fallback_events = []
    seasonal = args.seasonal_wind
    wind_months = None
    if seasonal:
        print(f"Loading 12 monthly NCEP wind snapshots (year {args.wind_year})...")
        try:
            wind_months = services.build_seasonal_wind_global(grid, year=args.wind_year)
            wind_src = f"seasonal cycle {args.wind_year} (12 monthly NCEP snapshots)"
        except Exception as e:
            if not args.allow_forcing_fallback:
                raise ValueError("requested seasonal wind unavailable; fallback disabled") from e
            fallback_events.append({"component": "wind", "reason": repr(e)})
            print(f"  seasonal wind fetch failed ({e!r}); fallback fixed-month")
            seasonal = False
    if not seasonal:
        y, m = (int(args.month[:4]), int(args.month[5:7]))
        month_idx = (y - 1948) * 12 + (m - 1)
        tau_x, tau_y = services.real_wind_forcing(month_idx=month_idx, grid=grid)
        wind_src = f"fixed {args.month} (NCEP R1)"
    T_sst = T_init[:, :, 0]
    lambda_bulk = 0.0 if args.no_bulk_flux else args.lambda_bulk * args.bulk_lambda_mult
    T_atm_source = "zonal WOA SST"
    T_atm = None
    T_atm_months = None
    if lambda_bulk > 0.0:
        if args.real_air_temp_monthly:
            if not seasonal:
                raise ValueError("--real-air-temp-monthly requires --seasonal-wind")
            try:
                T_atm_months = services.load_monthly_mean_air_temp(grid, year=args.wind_year)
                if args.ice_air_floor:
                    T_atm_months = np.maximum(np.asarray(T_atm_months), args.ice_air_floor_temp)
                    T_atm_source = f"monthly {args.wind_year} NCEP R1 2m air + {args.ice_air_floor_temp:g} C ice floor"
                if not args.ice_air_floor:
                    T_atm_source = f"monthly {args.wind_year} NCEP R1 2m air"
                if args.air_marine_smooth_passes > 0:
                    T_atm_months = np.stack(
                        [
                            marine_smooth_2d(f, grid.ocean_mask, args.air_marine_smooth_passes)
                            for f in T_atm_months
                        ]
                    )
                    T_atm_source += f" + {args.air_marine_smooth_passes} marine smooth"
                T_atm = np.mean(T_atm_months, axis=0)
            except Exception as exc:
                if not args.allow_forcing_fallback:
                    raise ValueError("requested NCEP air unavailable; fallback disabled") from exc
                fallback_events.append({"component": "air", "reason": repr(exc)})
                T_atm = None
                T_atm_source = "zonal WOA SST (explicit fallback)"
                print(
                    f"  WARNING: monthly NCEP air-temperature fetch failed ({exc!r}); falling back to zonal WOA SST target"
                )
                T_atm_months = None
        elif args.real_air_temp:
            try:
                T_atm = services.load_annual_mean_air_temp(grid, year=args.wind_year)
                T_atm_source = f"annual-mean {args.wind_year} NCEP R1 2m air"
                if args.ice_air_floor:
                    T_atm = np.maximum(np.asarray(T_atm), args.ice_air_floor_temp)
                    T_atm_source += f" + {args.ice_air_floor_temp:g} C ice floor"
                if args.air_marine_smooth_passes > 0:
                    T_atm = marine_smooth_2d(T_atm, grid.ocean_mask, args.air_marine_smooth_passes)
                    T_atm_source += f" + {args.air_marine_smooth_passes} marine smooth"
            except Exception as exc:
                if not args.allow_forcing_fallback:
                    raise ValueError("requested NCEP air unavailable; fallback disabled") from exc
                fallback_events.append({"component": "air", "reason": repr(exc)})
                T_atm = None
                T_atm_source = "zonal WOA SST (explicit fallback)"
                print(
                    f"  WARNING: real NCEP air-temperature fetch failed ({exc!r}); falling back to zonal WOA SST target"
                )
        if T_atm is None:
            T_atm = services.air_temp_profile(grid, T_sst)
        print(
            f"  bulk air-sea flux: lambda={lambda_bulk:.1f} W/m^2/K, T_atm={T_atm_source} ({float(np.nanmin(T_atm)):.2f}..{float(np.nanmax(T_atm)):.2f} C)"
        )
    S_sss = S_init[:, :, 0]
    S_ref_surf = None
    if args.sss_restore_days > 0.0:
        if args.sss_restore_zonal:
            prof = ocean_zonal_mean(grid, S_sss)
            S_ref_surf = np.broadcast_to(prof[None, :], (grid.nx, grid.ny)).copy()
            print(
                f"  SSS restoring: tau={args.sss_restore_days:g}d, target=ZONAL WOA SSS ({float(np.nanmin(prof)):.2f}..{float(np.nanmax(prof)):.2f} psu)"
            )
        else:
            S_ref_surf = S_sss
            print(
                f"  SSS restoring: tau={args.sss_restore_days:g}d, target=full 2D WOA SSS ({float(np.nanmin(S_sss)):.2f}..{float(np.nanmax(S_sss)):.2f} psu)"
            )
    if not seasonal:
        forcing_baked = (tau_x, tau_y, Q_heat)
    else:
        forcing_baked = (np.zeros_like(Q_heat), np.zeros_like(Q_heat), Q_heat)
    applied_arrays = {
        "initial_T": T_init,
        "initial_S": S_init,
        "baked": np.asarray(forcing_baked),
        "wind_months": wind_months,
        "air_months": T_atm_months,
        "T_atm": T_atm,
        "S_ref_surf": S_ref_surf,
    }
    for name, value in applied_arrays.items():
        if value is not None and (not np.isfinite(np.asarray(value, dtype=args.dtype)).all()):
            raise ValueError(f"nonfinite effective forcing/input: {name}")
    air_applied = (
        "disabled"
        if lambda_bulk <= 0
        else "monthly"
        if T_atm_months is not None
        else "annual"
        if args.real_air_temp and (not any((e["component"] == "air" for e in fallback_events)))
        else "zonal"
    )
    selected_files = services.input_files(args, seasonal=seasonal, air=air_applied)
    forcing_provenance = {
        "schema_version": 1,
        "requested": {
            "seasonal_wind": args.seasonal_wind,
            "month": args.month,
            "wind_year": args.wind_year,
            "annual_air": args.real_air_temp,
            "monthly_air": args.real_air_temp_monthly,
        },
        "applied": {"wind": wind_src, "air": T_atm_source if lambda_bulk > 0 else "disabled"},
        "fallback_events": fallback_events,
        "strict_local_inputs": args.strict_forcing,
        "selected_files": {
            name: {"path": str(path), "sha256": file_sha256(path) if path.is_file() else None}
            for name, path in selected_files.items()
        },
        "effective_arrays": fingerprint(
            {
                name: None if value is None else np.asarray(value, dtype=args.dtype)
                for name, value in applied_arrays.items()
            }
        ),
        "compute_dtype": args.dtype,
        "calendar": "repeating_360_day_30_day_months",
        "forcing_period_seconds": 360 * 86400 if seasonal else None,
        "time_origin": "January start of repeating climatological cycle"
        if seasonal
        else "fixed month",
        "interpolation": "bilinear space; 30-day months, linear boundary blend"
        if seasonal
        else "bilinear space; fixed time",
        "wind_blend_days": args.wind_blend_days,
        "wind_jit": args.wind_jit,
        "wind_stress_units": "N/m2",
        "prescribed_heat_units": "W/m2 into ocean",
        "air_units": "degC",
        "air_marine_smooth_passes": args.air_marine_smooth_passes,
        "ice_air_floor": args.ice_air_floor,
        "ice_air_floor_temp_C": args.ice_air_floor_temp,
        "restoration": {
            "sss_days": args.sss_restore_days,
            "sss_zonal": args.sss_restore_zonal,
            "coastal_days": args.coastal_restore_days,
            "global_sst_days": args.global_sst_restore_days,
        },
        "wind_spatial_processing": "bilinear clipped source coordinates; N/S taper",
        "prescribed_heat_source": "disabled"
        if args.no_meridional_heat_flux
        else "idealized_meridional_50_W_m2",
        "physical_forcing_qualification": "not_assessed",
    }
    source_identity = services.source_identity()
    execution_identity = {
        "python": sys.version,
        "jax": jax.__version__,
        "numpy": np.__version__,
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "flags": {
            name: os.environ.get(name)
            for name in (
                "XLA_FLAGS",
                "JAX_PLATFORMS",
                "JAX_ENABLE_X64",
                "JAX_COMPILATION_CACHE_DIR",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
            )
        },
    }
    return ForcingBundle(
        Q_heat=Q_heat,
        seasonal=seasonal,
        wind_months=wind_months,
        wind_src=wind_src,
        lambda_bulk=lambda_bulk,
        T_atm=T_atm,
        T_atm_months=T_atm_months,
        T_atm_source=T_atm_source,
        S_ref_surf=S_ref_surf,
        forcing_baked=forcing_baked,
        forcing_provenance=forcing_provenance,
        source_identity=source_identity,
        execution_identity=execution_identity,
    )
