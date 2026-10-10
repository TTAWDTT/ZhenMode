"""Frozen OMIP physical scope and plans; planning never launches a model."""

from __future__ import annotations

import argparse
import http.client
import json
import sys
from datetime import datetime
from pathlib import Path

from zhenmode.evaluation.protocols import digest, load_json
from zhenmode.provenance.sources import sha256_file

CONTRACT_ID = "zhenmode-omip2-physical-v1"
CASE_ID = "global-omip2-physical"
MECHANISMS = tuple(f"B{i:02d}" for i in range(1, 13))
PROFILES = {
    "surface-contract": (None, 0),
    "integration-6h": ("1958-01-01T06:00:00", 1),
    "integration-30d": ("1958-01-31T00:00:00", 1),
    "integration-1y": ("1959-01-01T00:00:00", 1),
    "climate-6cycle": ("2019-01-01T00:00:00", 6),
}


def contract():
    """One installed definition; data identities and effect thresholds are separate gates."""
    return {
        "schema_version": 1, "id": CONTRACT_ID, "case_id": CASE_ID,
        "scope": "global_ocean_sea_ice_climate",
        "calendar": "proleptic_gregorian", "interval": "left_closed_right_open",
        "forcing": {"product": "JRA55-do", "version": "1.4.0",
                    "first_year": 1958, "last_year": 2018,
                    "wind_height_m": 10, "temperature_humidity_height_m": 10,
                    "instant_sampling": "linear", "mean_sampling": "bounded_piecewise_constant",
                    "instant_cadence_s": 10800, "flux_cadence_s": 10800,
                    "runoff_cadence_s": 86400},
        "surface": {"bulk": "large_yeager_2009", "relative_wind_alpha": 1,
                    "moist_air": "gill_1982_jra_recommended",
                    "sst": "current_bulk", "minimum_scalar_wind_m_s": 0.5,
                    "coefficient_iteration": "five_max_relative_drag_1e-4",
                    "neutral_wind_floor_m_s": 0.3,
                    "air_potential_temperature": "Ta_plus_g_z_over_Gill_Cp",
                    "open_water_albedo": 0.066, "open_water_emissivity": 0.98,
                    "shortwave_penetration": "per_method_frozen_absorption_with_column_integral",
                    "direct_sst_restoring": False,
                    "sss_reference": "WOA13v2_upper_10m_monthly",
                    "sss_piston_m_s": 50 / (365 * 86400),
                    "sss_under_ice": True, "sss_global_mean_subtraction": False},
        "initial": {"temperature_salinity": "WOA13v2_annual", "velocity": "rest",
                    "eta_anomaly_m": 0, "ice_mass_kg_m2": 0},
        "geometry": {"scope": "global_including_poles_shallow_seas",
                     "bathymetry_source": "ETOPO2022_v1_r3600x1800_surface",
                     "native_discretization": "explicit_per_method",
                     "evaluation_grid": "1deg_360x180_centres"},
        "thermodynamics": "TEOS10_compatible_validated_nonlinear_T_S_p",
        "required_mechanisms": list(MECHANISMS),
        "evaluation": {"id": "omip2-climate-observations-v1",
                       "climate_cycle": 6, "main_window": [1980, 2009],
                       "ssh_window": [1993, 2009], "temporal_weighting": "interval_seconds",
                       "regridding": "separately_frozen_weights_and_masks",
                       "effect_thresholds": "not_declared"},
        "profiles": {name: {"start": "1958-01-01T00:00:00", "end": end, "cycles": cycles,
                             "climate_qualification": name == "climate-6cycle"}
                     for name, (end, cycles) in PROFILES.items()},
    }


def validate_frozen(value):
    expected = contract()
    envelope = {"contract": expected, "contract_sha256": digest(expected)}
    # Canonical bytes distinguish True/1 and 1/1.0, unlike Python dict equality.
    if not isinstance(value, dict) or digest(value) != digest(envelope):
        raise ValueError("benchmark contract differs from frozen v1; create a new version")
    return value


def plan(profile, method, dt_seconds):
    if profile not in PROFILES or method not in {"zhenmode", "mom6"}:
        raise ValueError("unknown benchmark profile or method")
    if type(dt_seconds) is not int or dt_seconds <= 0:
        raise ValueError("dt_seconds must be a positive integer in seconds")
    end, cycles = PROFILES[profile]
    duration = 0 if end is None else int((datetime.fromisoformat(end) - datetime(1958, 1, 1)).total_seconds()) * cycles
    if duration % dt_seconds:
        raise ValueError("dt_seconds must exactly divide the profile interval")
    definition = contract()
    return {"schema_version": 1, "contract_id": CONTRACT_ID,
            "contract_sha256": digest(definition), "case_id": CASE_ID,
            "profile": profile, "method": method, "dt_seconds": dt_seconds,
            "duration_seconds": duration, "steps": duration // dt_seconds,
            "physics_sha256": digest({key: value for key, value in definition.items()
                                      if key not in {"profiles", "evaluation"}}),
            "execution_status": "proposed", "execution_ready": False,
            "acceptance": "not_assessed", "comparability": "not_yet_comparable",
            "cost_estimate": "requires_same_physics_measurement",
            "required_gates": ["input_receipts", "B01-B12_actual_configuration",
                               "build_receipt", "diagnostic_contract", "resource_plan"],
            "climate_qualification": False,
            "purpose": "component_contract" if profile == "surface-contract" else
                       "full_climate_experiment" if profile == "climate-6cycle" else
                       "short_subset_same_physics"}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "standing-wave":
        from zhenmode.execution.standing_wave import main as wave
        return wave(arguments[1:])
    if arguments and arguments[0] == "channel":
        from zhenmode.execution.channel_dynamics import main as channel
        return channel(arguments[1:])
    if arguments and arguments[0] == "prepare-wind-sample":
        from zhenmode.preparation.wind_sample import main as wind_sample
        return wind_sample(arguments[1:])
    if arguments and arguments[0] == "forced-channel":
        from zhenmode.execution.forced_channel import main as forced
        return forced(arguments[1:])
    parser = argparse.ArgumentParser(prog="zhenmode benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("standing-wave", help="run native full models on one frozen Cartesian wave")
    commands.add_parser("channel", help="run rotating adjustment or stratified thermal-wind benchmarks")
    commands.add_parser("prepare-wind-sample", help="derive a verified real-weather stress sample")
    commands.add_parser("forced-channel", help="run a prescribed real-weather stress control")
    commands.add_parser("describe", help="show the installed frozen physical definition")
    freeze = commands.add_parser("freeze", help="create a contract receipt, never overwrite")
    freeze.add_argument("--output", required=True)
    check = commands.add_parser("check-contract")
    check.add_argument("--contract", required=True)
    prepare = commands.add_parser("plan", help="expand a profile without launching anything")
    prepare.add_argument("--profile", required=True, choices=tuple(PROFILES))
    prepare.add_argument("--method", required=True, choices=("zhenmode", "mom6"))
    prepare.add_argument("--dt-seconds", required=True, type=int)
    forcing = commands.add_parser("check-forcing", help="verify native CF files/window and sample the first interval")
    forcing.add_argument("--manifest", required=True)
    forcing.add_argument("--grid", required=True, help="NPZ containing lon, lat, wet_mask")
    forcing.add_argument("--start", required=True, help="Gregorian ISO timestamp, no timezone suffix")
    forcing.add_argument("--end", required=True)
    fetch = commands.add_parser('fetch-jra', help='download published original annual files with SHA256 checks; never launch a model')
    fetch.add_argument('--catalog', required=True, help='complete ESGF File search response JSON')
    fetch.add_argument('--destination', required=True)
    fetch.add_argument('--year', type=int, required=True)
    native = commands.add_parser('prepare-forcing', help='prepare an explicit native rectangular JRA window; never launch an ocean')
    native.add_argument('--acquisition', action='append', required=True,
                        help='repeat for adjacent verified annual shards, including the end-window instant')
    native.add_argument('--grid', required=True, help='NPZ: lon/lat, lon_bounds/lat_bounds, area, wet_mask')
    native.add_argument('--runoff-area', required=True, help='JSON: path, variable, bytes, sha256 of original discharge grid-cell area')
    native.add_argument('--output', required=True)
    native.add_argument('--start', required=True)
    native.add_argument('--end', required=True)
    native.add_argument('--maximum-routing-distance-m', type=float, default=500000)
    initial=commands.add_parser('prepare-initial-source',help='convert original WOA annual T/SP to PT/CT/SR; preserve missing support')
    initial.add_argument('--acquisition',required=True)
    initial.add_argument('--pressure-reference',required=True,help='identity-bound pressure JSON, p[depth,lat] in dbar')
    initial.add_argument('--output',required=True)
    native_initial=commands.add_parser('prepare-native-initial',help='prepare fixed native WOA points; unresolved support blocks eligibility')
    native_initial.add_argument('--source-prepared',required=True)
    native_initial.add_argument('--geometry',required=True)
    native_initial.add_argument('--nodes-file',required=True,help='JSON containing z_nodes_m, negative downward')
    native_initial.add_argument('--output',required=True)
    geometry=commands.add_parser('prepare-native-geometry',help='prepare binary coastal geometry with identity-bound shoreline and channel evidence')
    geometry.add_argument('--parent-geometry',required=True)
    geometry.add_argument('--shoreline-acquisition',required=True)
    geometry.add_argument('--policy',required=True)
    geometry.add_argument('--output',required=True)
    bottom=commands.add_parser('complete-native-bottom',help='complete reviewed native bottom gaps; execution qualification remains separate')
    bottom.add_argument('--native-prepared',required=True)
    bottom.add_argument('--review-file',required=True)
    bottom.add_argument('--output',required=True)
    fd_initial=commands.add_parser('prepare-fd-initial',help='prepare explicit fixed-partial CT/SR/pressure and area metrics for the FD factory; no integration')
    fd_initial.add_argument('--native-prepared',required=True)
    fd_initial.add_argument('--policy',required=True)
    fd_initial.add_argument('--output',required=True)
    tripolar=commands.add_parser('prepare-tripolar-grid',help='prepare identity-bound MOM angular-grid/metric candidate; no mask, mapping or model run')
    tripolar.add_argument('--acquisition',required=True)
    tripolar.add_argument('--south-boundary',required=True,type=float)
    tripolar.add_argument('--output',required=True)
    wind_run=commands.add_parser('run-fd-wind',help='actual bounded GPU wind-driven component with saved budgets/restart; heat/freshwater/ice remain inactive')
    wind_run.add_argument('--native-prepared',required=True)
    wind_run.add_argument('--forcing-manifest',required=True)
    wind_run.add_argument('--output',required=True)
    wind_run.add_argument('--start',default='1958-01-01T00:00:00')
    wind_run.add_argument('--dt-seconds',type=float,required=True)
    wind_run.add_argument('--steps',type=int,required=True)
    wind_run.add_argument('--wall-seconds',type=int,default=360)
    wind_run.add_argument('--resume')
    wind_run.add_argument('--polar-cap-rows',type=int,default=2)
    wind_run.add_argument('--polar-cap-taper',type=int,default=3)
    wind_run.add_argument('--match-transport',action='store_true')
    sis_build=commands.add_parser('compile-sis2-bridge',help='link native SIS2 exchange against a verified coupled object build; no ocean integration')
    sis_build.add_argument('--coupled-build',required=True)
    sis_build.add_argument('--output',required=True)
    sis_case=commands.add_parser('prepare-sis2-case',help='prepare native sea-ice supergrid and topography matching approved FD inputs; no integration')
    sis_case.add_argument('--native-prepared',required=True)
    sis_case.add_argument('--output',required=True)
    sis_case.add_argument('--start',default='1958-01-01T00:00:00')
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare-tripolar-grid':
            from zhenmode.baselines.mom6.tripolar import prepare_tripolar_grid
            result=prepare_tripolar_grid(args.acquisition,args.south_boundary,args.output)
        elif args.command == 'prepare-sis2-case':
            from zhenmode.coupling.geometry import prepare_sis2_case
            result=prepare_sis2_case(args.native_prepared,args.output,start=args.start)
        elif args.command == 'compile-sis2-bridge':
            from zhenmode.coupling.sis2 import compile_sis2_bridge
            result=compile_sis2_bridge(args.coupled_build,args.output)
        elif args.command == 'run-fd-wind':
            from zhenmode.execution.wind_run import run_fd_wind
            result=run_fd_wind(args.native_prepared,args.forcing_manifest,args.output,start=args.start,
                dt_seconds=args.dt_seconds,steps=args.steps,wall_seconds=args.wall_seconds,resume=args.resume,
                polar_cap_rows=args.polar_cap_rows,polar_cap_taper=args.polar_cap_taper,match_transport=args.match_transport)
        elif args.command == 'prepare-fd-initial':
            from zhenmode.preparation.fd import prepare_fd_native_inputs
            result=prepare_fd_native_inputs(args.native_prepared,args.policy,args.output)
        elif args.command == 'complete-native-bottom':
            from zhenmode.preparation.bottom import complete_native_bottom
            result=complete_native_bottom(args.native_prepared,args.review_file,args.output)
        elif args.command == 'prepare-native-geometry':
            from zhenmode.preparation.coast import prepare_native_geometry
            result=prepare_native_geometry(args.parent_geometry,args.shoreline_acquisition,args.policy,args.output)
        elif args.command == 'prepare-native-initial':
            from zhenmode.preparation.native_initial import prepare_native_initialization
            result=prepare_native_initialization(args.source_prepared,args.geometry,args.nodes_file,args.output)
        elif args.command == 'prepare-initial-source':
            from zhenmode.preparation.woa import prepare_woa_thermodynamics
            result=prepare_woa_thermodynamics(args.acquisition,args.pressure_reference,args.output)
        elif args.command == 'prepare-forcing':
            from zhenmode.preparation.forcing import prepare_jra_window

            origin = datetime(1970, 1, 1)
            result = prepare_jra_window(args.acquisition, args.grid, args.runoff_area, args.output,
                start=(datetime.fromisoformat(args.start)-origin).total_seconds(),
                end=(datetime.fromisoformat(args.end)-origin).total_seconds(),
                maximum_routing_distance_m=args.maximum_routing_distance_m)
        elif args.command == 'fetch-jra':
            from zhenmode.preparation.acquisition import fetch_jra

            result = fetch_jra(args.catalog, args.destination, args.year)
        elif args.command == "check-forcing":
            import numpy as np

            from zhenmode.model.inputs.forcing.jra55 import JRA55Forcing

            origin = datetime(1970, 1, 1)
            start = (datetime.fromisoformat(args.start) - origin).total_seconds()
            end = (datetime.fromisoformat(args.end) - origin).total_seconds()
            with np.load(args.grid, allow_pickle=False) as grid:
                if set(grid.files) != {'lon', 'lat', 'wet_mask'}:
                    raise ValueError('forcing grid NPZ must contain exactly lon, lat, wet_mask')
                reader = JRA55Forcing(args.manifest, lon=grid['lon'], lat=grid['lat'],
                                     wet_mask=grid['wet_mask'], start_seconds=start, end_seconds=end)
            first_end = min(end, start + 10800)
            sample = reader.sample(start, interval_end_seconds=first_end)
            result = {'status': 'metadata_identity_window_and_first_sample_verified',
                      'data_kind': reader.data_kind, 'file_receipts': reader.receipts,
                      'grid_sha256': sha256_file(args.grid),
                      'sample_interval_seconds': [start, first_end],
                      'all_records_values_scanned': False,
                      'first_sample_wet_ranges': {name: [float(np.min(np.asarray(field)[reader.wet])),
                                                        float(np.max(np.asarray(field)[reader.wet]))]
                                                  for name, field in zip(sample._fields, sample, strict=True)},
                      'execution_ready': False, 'climate_qualification': False}
        elif args.command == "plan":
            result = plan(args.profile, args.method, args.dt_seconds)
        elif args.command == "check-contract":
            validate_frozen(load_json(args.contract))
            result = {"status": "physical_contract_verified", "file_sha256": sha256_file(args.contract),
                      "execution_ready": False, "climate_qualification": False}
        else:
            definition = contract()
            result = {"contract": definition, "contract_sha256": digest(definition)}
            if args.command == "freeze":
                with Path(args.output).open("x", encoding="utf-8") as stream:
                    json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
                    stream.write("\n")
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        if args.command=='run-fd-wind' and result['status']=='failed':
            return 1
        if args.command in ('prepare-native-initial','complete-native-bottom') and not result['complete_wet_support']:
            return 3
        if args.command == 'prepare-native-geometry' and result['status'] != 'native_geometry_prepared':
            return 3
        return 0
    except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException) as error:
        parser.error(str(error))
