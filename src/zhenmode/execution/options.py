"""Use the production parser as the only owner of accepted options and defaults."""

from __future__ import annotations

import argparse
import os

from .schema import ConfigurationError, number, quantity

# Canonical units are deliberately exact. There is no implicit unit conversion.
UNITS = {
    "dt": "s", "dt_bt": "s", "nu_h": "m2/s", "nu_bi": "m4/s",
    "kappa_v": "m2/s", "kappa_conv": "m2/s", "kappa_gm": "m2/s",
    "kappa_redi": "m2/s", "lambda_bulk": "W/m2/K",
    "min_depth": "m", "ice_air_floor_temp": "degC", "ice_freeze_temp": "degC",
    "ice_salt_flux": "psu/s", "ice_insulation_scale_m": "m",
    "sss_restore_days": "day", "coastal_restore_days": "day",
    "global_sst_restore_days": "day", "coastal_bulk_lambda": "W/m2/K",
    "coastal_kappa_h": "m2/s", "coastal_kappa_v": "m2/s",
    "mixed_layer_depth": "m", "mixed_layer_depth_min": "m",
    "mixed_layer_depth_max": "m", "mld_density_delta": "kg/m3",
    "sponge_days": "day", "eta_relax_days": "day",
    "eta_relax_buffer": "deg", "wind_blend_days": "day", "checkpoint_days": "day",
}
VECTOR_UNITS = {"mixed_layer_lat_band": "deg", "eta_relax_box": "deg"}
CASE_OPTIONS = {
    "days", "snap_days", "lat_max", "resolution", "resolution_remap", "ny", "z_levels",
    "seasonal_wind", "wind_year", "month", "real_air_temp", "real_air_temp_monthly",
    "no_bulk_flux", "no_meridional_heat_flux", "global_sst_restore_days", "sss_restore_days",
}
CONTROLLED_OPTIONS = {
    "tag", "out_dir", "log_dir", "init_from", "restart_from", "strict_forcing",
    "allow_forcing_fallback", "max_steps",
}


def production_parser():
    from zhenmode.model.runtime.cli import build_run_parser

    return build_run_parser()


def option_actions():
    return {action.dest: action for action in production_parser()._actions if action.dest != "help"}


def decode_options(options, *, common=False):
    if not isinstance(options, dict):
        raise ConfigurationError("options must be a mapping")
    actions = option_actions()
    result = {}
    for name, value in options.items():
        if name not in actions:
            raise ConfigurationError(f"unknown production option: {name}")
        if name in CONTROLLED_OPTIONS or (name in CASE_OPTIONS and not common):
            raise ConfigurationError(f"{name} is owned by the case or run manager")
        if name in UNITS:
            value = quantity(value, UNITS[name], name)
        elif name in VECTOR_UNITS:
            value = quantity(value, VECTOR_UNITS[name], name, vector=True)
        action = actions[name]
        if isinstance(action, argparse._StoreTrueAction):
            if not isinstance(value, bool):
                raise ConfigurationError(f"{name} must be boolean")
        elif action.type is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigurationError(f"{name} must be an integer")
        elif action.type is float and name not in VECTOR_UNITS:
            number(value, name)
        elif action.type is None and action.choices is None:
            if value is not None and not isinstance(value, str):
                raise ConfigurationError(f"{name} must be a string")
        if action.choices is not None and value not in action.choices:
            raise ConfigurationError(f"{name}: expected one of {list(action.choices)}")
        result[name] = value
    return result


def argv_for(options):
    actions = option_actions()
    arguments = []
    for name, value in sorted(options.items()):
        if value is None:
            continue
        action = actions[name]
        flag = next(flag for flag in action.option_strings if flag.startswith("--"))
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                arguments.append(flag)
        elif action.nargs in (2, 4):
            arguments.extend([flag, *(str(item) for item in value)])
        else:
            arguments.extend([flag, str(value)])
    return arguments


def resolved_runtime(options):
    """Expand parser, physics, environment-dependent and resolution defaults."""
    from zhenmode.model.runtime.cli import (
        DT_BT_DEFAULT,
        NU_BI_DEFAULT,
        NU_H_DEFAULT,
        NY_DEFAULT,
        _validate_arguments,
        scaled_physics_for_resolution,
    )

    parser = production_parser()
    try:
        args = parser.parse_args(argv_for(options))
        requested_steps = _validate_arguments(args)
    except (SystemExit, ValueError) as error:
        raise ConfigurationError(f"invalid production options: {error}") from error
    if args.resolution is not None:
        args.dt_bt, args.nu_h, args.nu_bi = scaled_physics_for_resolution(
            args.resolution, args.dt_bt, args.nu_h, args.nu_bi
        )
    else:
        args.dt_bt = DT_BT_DEFAULT if args.dt_bt is None else args.dt_bt
        args.nu_h = NU_H_DEFAULT if args.nu_h is None else args.nu_h
        args.nu_bi = NU_BI_DEFAULT if args.nu_bi is None else args.nu_bi
    args.kappa_v = 1e-5 if args.kappa_v is None else args.kappa_v
    # Freeze the actual environment default rather than let it drift at launch.
    if args.projection_niter is None:
        try:
            args.projection_niter = int(os.environ.get("OCEAN_PAV_NITER", "150"))
        except ValueError as error:
            raise ConfigurationError("OCEAN_PAV_NITER must be an integer") from error
        if args.projection_niter <= 0:
            raise ConfigurationError("OCEAN_PAV_NITER must be positive")
    if args.resolution is None:
        args.ny = NY_DEFAULT if args.ny is None else args.ny
    args.strict_forcing = True
    if args.real_air_temp_monthly and not args.seasonal_wind:
        raise ConfigurationError("monthly air forcing requires seasonal wind")
    if args.mixed_layer_mode == "stratification" and args.mixed_layer_depth is None:
        raise ConfigurationError("stratification MLD requires mixed_layer_depth")
    if args.global_sst_restore_days and args.coastal_restore_days:
        raise ConfigurationError("global and coastal SST restoring are mutually exclusive")
    for name in ("days", "snap_days", "checkpoint_days"):
        seconds = getattr(args, name) * 86400.0
        if seconds and abs(seconds / args.dt - round(seconds / args.dt)) > 1e-8:
            raise ConfigurationError(f"{name} must align exactly with dt")
    if args.snap_days > args.days:
        raise ConfigurationError("snapshot interval exceeds duration")
    result = vars(args)
    for name in CONTROLLED_OPTIONS - {"strict_forcing", "allow_forcing_fallback", "max_steps"}:
        result.pop(name)
    return result, requested_steps
