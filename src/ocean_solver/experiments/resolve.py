"""Resolve a common problem, method preset, and explicit experiment changes."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import asdict
from pathlib import Path

from .options import CASE_OPTIONS, decode_options, resolved_runtime
from .schema import (
    ConfigurationError,
    fields,
    identifier,
    load_document,
    quantity,
    reference_path,
    text_value,
)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_case(path):
    case = load_document(path)
    fields(case, ("schema_version", "kind", "id", "name_zh", "description", "domain", "grid", "duration", "output_interval", "initial", "forcing", "data", "comparison"), where="case")
    if case["kind"] != "case":
        raise ConfigurationError("expected kind: case")
    identifier(case["id"])
    text_value(case["name_zh"], "name_zh")
    text_value(case["description"], "description")
    fields(case["domain"], ("boundary", "lat_max", "coriolis"), where="domain")
    if case["domain"]["boundary"] != "periodic_lon_closed_lat":
        raise ConfigurationError("production case requires periodic longitude and closed latitude walls")
    if case["domain"]["coriolis"] not in {"rotating_earth", "none"}:
        raise ConfigurationError("unknown coriolis setting")
    lat_max = quantity(case["domain"]["lat_max"], "deg", "lat_max", positive=True)
    if lat_max >= 90:
        raise ConfigurationError("lat_max must be below 90 degrees")
    grid = fields(case["grid"], ("kind", "resolution", "resolution_remap", "z_levels"), ("nx", "ny", "depth"), where="grid")
    if grid["kind"] not in {"etopo", "synthetic"}:
        raise ConfigurationError("unknown grid kind")
    resolution = quantity(grid["resolution"], "deg", "resolution", positive=True)
    if grid["resolution_remap"] not in {"legacy", "area"}:
        raise ConfigurationError("unknown resolution_remap")
    levels = quantity(grid["z_levels"], "m", "z_levels", vector=True)
    if len(levels) < 3 or levels[0] != 0 or any(a <= b for a, b in zip(levels, levels[1:])):
        raise ConfigurationError("z_levels must start at zero and strictly descend")
    if grid["kind"] == "synthetic":
        for name in ("nx", "ny"):
            if isinstance(grid.get(name), bool) or not isinstance(grid.get(name), int) or grid[name] < 4:
                raise ConfigurationError(f"synthetic grid requires integer {name} >= 4")
        depth = quantity(grid.get("depth"), "m", "depth", positive=True)
        if depth < -levels[-1]:
            raise ConfigurationError("synthetic all-wet depth cannot be shallower than bottom level")
        if abs(resolution - 360.0 / grid["nx"]) > 1e-10:
            raise ConfigurationError("synthetic longitude resolution must equal 360/nx")
    elif case["domain"]["coriolis"] != "rotating_earth":
        raise ConfigurationError("ETOPO production grid uses rotating-earth coriolis")
    duration = quantity(case["duration"], "s", "duration", positive=True)
    interval = quantity(case["output_interval"], "s", "output_interval", positive=True)
    if interval > duration:
        raise ConfigurationError("output interval exceeds duration")
    initial = fields(case["initial"], ("kind",), ("temperature", "salinity"), where="initial")
    if initial["kind"] == "uniform":
        quantity(initial.get("temperature"), "degC", "initial temperature")
        quantity(initial.get("salinity"), "psu", "initial salinity", positive=True)
        if grid["kind"] != "synthetic":
            raise ConfigurationError("uniform initial fields are supported only by synthetic case services")
    elif initial["kind"] != "woa2023" or set(initial) != {"kind"}:
        raise ConfigurationError("initial must be woa2023 or explicit uniform fields")
    if grid["kind"] == "synthetic" and initial["kind"] != "uniform":
        raise ConfigurationError("synthetic case requires uniform initial fields")
    forcing = fields(case["forcing"], ("kind", "description", "options"), ("monthly_tau_increment",), where="forcing")
    if forcing["kind"] not in {"seasonal_ncep2023", "wind_only", "restoring", "synthetic_monthly"}:
        raise ConfigurationError("unknown forcing kind")
    options = decode_options(forcing["options"], common=True)
    if options.get("real_air_temp") and options.get("real_air_temp_monthly"):
        raise ConfigurationError("annual and monthly real air forcing are ambiguous together")
    if options.get("no_bulk_flux") and (options.get("real_air_temp") or options.get("real_air_temp_monthly")):
        raise ConfigurationError("disabled bulk heat flux cannot also request real air input")
    if forcing["kind"] == "seasonal_ncep2023" and (options.get("seasonal_wind") is not True or options.get("wind_year") != 2023):
        raise ConfigurationError("seasonal_ncep2023 requires seasonal wind from year 2023")
    if forcing["kind"] in {"wind_only", "restoring"} and (options.get("no_bulk_flux") is not True or options.get("no_meridional_heat_flux") is not True):
        raise ConfigurationError("wind_only/restoring cases require bulk and idealized heat flux disabled")
    if forcing["kind"] == "wind_only" and (options.get("sss_restore_days", 0) or options.get("global_sst_restore_days", 0)):
        raise ConfigurationError("wind_only cannot restore surface temperature or salinity")
    if forcing["kind"] == "restoring" and (options.get("sss_restore_days", 0) <= 0 or options.get("global_sst_restore_days", 0) <= 0):
        raise ConfigurationError("restoring case requires positive surface temperature and salinity restoration")
    allowed_forcing = CASE_OPTIONS - {"days", "snap_days", "lat_max", "resolution", "resolution_remap", "ny", "z_levels"}
    if set(options) - allowed_forcing:
        raise ConfigurationError("case forcing options must describe shared physical forcing")
    if forcing["kind"] == "synthetic_monthly":
        quantity(forcing.get("monthly_tau_increment"), "N/m2", "monthly_tau_increment")
        if grid["kind"] != "synthetic" or not options.get("no_bulk_flux") or not options.get("no_meridional_heat_flux"):
            raise ConfigurationError("synthetic service supports no bulk/idealized heat flux only")
    if grid["kind"] == "synthetic" and (forcing["kind"] != "synthetic_monthly" or options.get("seasonal_wind") is not True):
        raise ConfigurationError("synthetic case requires synthetic_monthly and seasonal_wind=true")
    if not isinstance(case["data"], list):
        raise ConfigurationError("data must be a list of references")
    roles = set()
    for data in case["data"]:
        fields(data, ("role", "path", "version", "sha256", "status", "source"), where="data reference")
        identifier(data["role"], "data role")
        if data["role"] in roles:
            raise ConfigurationError(f"duplicate data role: {data['role']}")
        roles.add(data["role"])
        for name in ("path", "version", "source"):
            text_value(data[name], f"data.{name}")
        if data["status"] not in {"available", "pending"}:
            raise ConfigurationError("data status must be available or pending")
        sha = data["sha256"]
        if sha is not None and (not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
            raise ConfigurationError("data.sha256 must be a lowercase SHA256 or null with pending status")
        if sha is None and data["status"] != "pending":
            raise ConfigurationError("available data requires an expected SHA256")
    if grid["kind"] == "synthetic" and case["data"]:
        raise ConfigurationError("synthetic case generates its inputs; external data are unsupported")
    if grid["kind"] == "etopo":
        required = {"bathymetry", "temperature", "salinity"}
        required |= {f"wind-{month:02d}" for month in range(1, 13)} if options.get("seasonal_wind") else {"wind-fixed"}
        if options.get("real_air_temp") or options.get("real_air_temp_monthly"):
            required.add("air")
        if roles != required:
            raise ConfigurationError(f"data roles do not cover exactly selected inputs: missing={sorted(required - roles)} unexpected={sorted(roles - required)}")
    comparison = fields(case["comparison"], ("scope", "limitations"), where="comparison")
    text_value(comparison["scope"], "comparison scope")
    if not isinstance(comparison["limitations"], list) or any(not isinstance(x, str) for x in comparison["limitations"]):
        raise ConfigurationError("comparison limitations must be strings")
    options |= {"days": duration / 86400.0, "snap_days": interval / 86400.0, "lat_max": lat_max, "resolution": resolution,
                "resolution_remap": grid["resolution_remap"], "z_levels": ",".join(str(x) for x in levels)}
    return case, options


def apply_changes(options, changes):
    if not isinstance(changes, list):
        raise ConfigurationError("changes must be a list")
    result = dict(options)
    seen = set()
    normalized = []
    for change in changes:
        fields(change, ("option", "before", "after", "factor", "reason"), where="change")
        name = text_value(change["option"], "change option")
        if name in seen:
            raise ConfigurationError(f"ambiguous repeated override: {name}")
        seen.add(name)
        before = decode_options({name: change["before"]})[name]
        after = decode_options({name: change["after"]})[name]
        if name not in result or result[name] != before:
            raise ConfigurationError(f"{name}: declared before {before!r} does not match resolved parent {result.get(name)!r}")
        if before == after:
            raise ConfigurationError(f"{name}: change does not change a value")
        text_value(change["factor"], "factor")
        text_value(change["reason"], "reason")
        result[name] = after
        normalized.append({**change, "before_value": before, "after_value": after})
    return result, normalized


def read_preset(path, root, stack=()):
    path = Path(path).resolve()
    if path in stack:
        raise ConfigurationError("cyclic preset inheritance: " + " -> ".join(str(x) for x in (*stack, path)))
    preset = load_document(path)
    fields(preset, ("schema_version", "kind", "id", "name_zh", "method", "purpose", "source", "discretization", "options"), ("parent", "overrides"), where="preset")
    if preset["kind"] != "preset" or preset["method"] != "zhenmode":
        raise ConfigurationError("this preset resolver runs method zhenmode; MOM6 uses its baseline contract")
    identifier(preset["id"])
    for name in ("name_zh", "purpose", "source", "discretization"):
        text_value(preset[name], name)
    options = decode_options(preset["options"])
    lineage = [{"path": str(path), "id": preset["id"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]
    if preset.get("parent"):
        _, inherited, inherited_lineage = read_preset(reference_path(root, preset["parent"]), root, (*stack, path))
        if set(inherited) & set(options):
            raise ConfigurationError("inherited values must use explicit before/after overrides")
        inherited, _ = apply_changes(inherited, preset.get("overrides", []))
        options = inherited | options
        lineage = inherited_lineage + lineage
    elif preset.get("overrides"):
        raise ConfigurationError("preset overrides require a parent")
    return preset, options, lineage


def effective_grid(case):
    grid = case["grid"]
    if grid["kind"] == "synthetic":
        return {"nx": grid["nx"], "ny": grid["ny"], "nz": len(grid["z_levels"]["value"]), "kind": "synthetic"}
    from ocean_solver.io.grid import global_grid_dims

    nx, ny = global_grid_dims(grid["resolution"]["value"], case["domain"]["lat_max"]["value"], remap=grid["resolution_remap"])
    if "nx" in grid and grid["nx"] != nx or "ny" in grid and grid["ny"] != ny:
        raise ConfigurationError(f"declared grid dimensions disagree with production remap: {nx}x{ny}")
    return {"nx": nx, "ny": ny, "nz": len(grid["z_levels"]["value"]), "kind": "etopo"}


def expand_experiment(path, root=None, *, document=None):
    root = Path(root or Path.cwd()).resolve()
    path = reference_path(root, str(path))
    experiment = load_document(path) if document is None else document
    fields(experiment, ("schema_version", "kind", "id", "name_zh", "purpose", "method", "case", "parent_preset", "variant", "changes", "evaluation_protocol", "protocol_path"), ("resources",), where="experiment")
    if type(experiment["schema_version"]) is not int or experiment["schema_version"] != 1 or experiment["kind"] != "experiment" or experiment["method"] != "zhenmode":
        raise ConfigurationError("expected v1 zhenmode experiment")
    identifier(experiment["id"])
    for name in ("name_zh", "purpose", "evaluation_protocol", "protocol_path"):
        text_value(experiment[name], name)
    variant = experiment["variant"]
    if variant not in {"baseline", "single_factor", "combination", "tuning"}:
        raise ConfigurationError("variant must be baseline, single_factor, combination or tuning")
    case_path = reference_path(root, experiment["case"])
    case, common = read_case(case_path)
    preset, options, lineage = read_preset(reference_path(root, experiment["parent_preset"]), root)
    parent, _ = resolved_runtime(common | options)
    runtime, changes = apply_changes(parent, experiment["changes"])
    if variant == "baseline" and changes:
        raise ConfigurationError("baseline cannot declare changes")
    if variant == "single_factor" and len(changes) != 1:
        raise ConfigurationError("single_factor requires exactly one changed production option")
    if variant == "combination" and len(changes) < 2:
        raise ConfigurationError("combination requires at least two changed options")
    if variant == "tuning" and not changes:
        raise ConfigurationError("tuning requires declared changes")
    runtime, steps = resolved_runtime(runtime)
    actual = {name for name in parent if parent[name] != runtime[name]}
    if actual != {change["option"] for change in changes}:
        raise ConfigurationError(f"undeclared effective option changes: {sorted(actual)}")
    resources = experiment.get("resources", {"cpu": 1, "wall_seconds": 180, "memory_mib": 4096})
    fields(resources, ("cpu", "wall_seconds", "memory_mib"), where="resources")
    for name in resources:
        if isinstance(resources[name], bool) or not isinstance(resources[name], int) or resources[name] <= 0:
            raise ConfigurationError("resources must be positive integers")
    from ocean_solver.config.definitions import PhysicsConfig

    physics = asdict(PhysicsConfig())
    for name in ("nu_h", "nu_bi", "kappa_v", "kappa_conv", "kappa_gm", "kappa_redi", "gm_slope_max"):
        physics[name] = runtime[name]
    physics["kappa_bi"] = runtime["nu_bi"]
    grid = effective_grid(case)
    duration = case["duration"]["value"]
    interval = case["output_interval"]["value"]
    count = int(duration // interval)
    output_times = [i * interval for i in range(count + 1)]
    if output_times[-1] != duration:
        output_times.append(duration)
    protocol_path = reference_path(root, experiment["protocol_path"])
    if not protocol_path.is_file():
        raise ConfigurationError(f"missing evaluation protocol: {protocol_path}")
    from ocean_solver.evaluation.protocols import load_protocol

    try:
        protocol = load_protocol(protocol_path)
    except ValueError as error:
        raise ConfigurationError(f"invalid evaluation protocol: {error}") from error
    if protocol.get("id") != experiment["evaluation_protocol"] or protocol.get("case_id") != case["id"]:
        raise ConfigurationError("protocol identity or case does not match experiment")
    if not math.isclose(protocol["window"]["end_day"] * 86400, duration, rel_tol=0, abs_tol=1e-8):
        raise ConfigurationError("protocol scoring-window endpoint must match complete case duration")
    result = {
        "schema_version": 1, "experiment_id": experiment["id"], "name_zh": experiment["name_zh"],
        "purpose": experiment["purpose"], "method": "zhenmode", "case_id": case["id"],
        "preset_id": preset["id"], "variant": variant, "changes": changes,
        "case": case, "duration_s": duration, "output_times_s": output_times,
        "runtime_options": runtime, "effective_physics": physics, "effective_grid": grid,
        "effective_parameterizations": {name: runtime[name] for name in (
            "lambda_bulk", "bulk_lambda_mult", "ice_air_floor", "ice_air_floor_temp", "ice_salt_flux", "ice_freeze_temp",
            "dynamic_ice", "ice_insulation_scale_m", "mixed_layer_depth", "mixed_layer_mode", "mld_density_delta",
            "mixed_layer_depth_min", "mixed_layer_depth_max", "mixed_layer_lat_band", "sss_restore_zonal",
            "coastal_restore_days", "coastal_restore_cells", "coastal_restore_taper", "coastal_bulk_lambda",
            "coastal_bulk_cells", "coastal_kappa_h", "coastal_kappa_h_cells", "coastal_kappa_v", "coastal_kappa_v_cells",
            "sponge_days", "sponge_cells", "eta_relax_days", "eta_relax_box", "eta_relax_buffer")},
        "effective_timestepping": {"barotropic_substeps": max(1, round(runtime["dt"] / runtime["dt_bt"])) if runtime["mode_split"] else 0,
                                    "barotropic_dt_s": runtime["dt"] / max(1, round(runtime["dt"] / runtime["dt_bt"])) if runtime["mode_split"] else 0},
        "requested_steps": steps, "resources": resources,
        "evaluation_protocol": experiment["evaluation_protocol"], "protocol": protocol,
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "protocol_file_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "protocol_content_sha256": canonical_hash(protocol),
        "provenance": {"experiment_path": str(path), "case_path": str(case_path), "preset_lineage": lineage,
                       "protocol_path": str(protocol_path),
                       "case_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest()},
    }
    # Changing a human title, purpose, or sweep ID does not change model config.
    result["config_hash"] = canonical_hash({
        "method": result["method"], "runtime_options": runtime,
        "case": {name: case[name] for name in ("domain", "grid", "initial", "forcing", "duration", "output_interval", "data")},
        "physics": physics, "parameterizations": result["effective_parameterizations"],
        "grid": grid, "timestepping": result["effective_timestepping"],
    })
    return result


def resource_estimate(expanded):
    grid = expanded["effective_grid"]
    cells = grid["nx"] * grid["ny"] * grid["nz"]
    bytes_per_number = 4 if expanded["runtime_options"]["dtype"] == "float32" else 8
    return {"grid_cells": cells, "steps": expanded["requested_steps"],
            "state_arrays_lower_bound_bytes": 6 * cells * bytes_per_number,
            "cell_steps": cells * expanded["requested_steps"], "wall_seconds_estimate": None,
            "estimate_status": "uncalibrated; array lower bound excludes compiler, work arrays and IO",
            "limits": expanded["resources"]}


def expand_sweep(path, root=None):
    root = Path(root or Path.cwd()).resolve()
    sweep = load_document(reference_path(root, str(path)))
    fields(sweep, ("schema_version", "kind", "id", "name_zh", "purpose", "experiment", "axes", "max_runs"), where="sweep")
    if sweep["kind"] != "sweep":
        raise ConfigurationError("expected kind: sweep")
    identifier(sweep["id"])
    text_value(sweep["name_zh"], "name_zh")
    text_value(sweep["purpose"], "purpose")
    if not isinstance(sweep["axes"], list) or not sweep["axes"]:
        raise ConfigurationError("sweep needs at least one axis")
    if isinstance(sweep["max_runs"], bool) or not isinstance(sweep["max_runs"], int) or not 1 <= sweep["max_runs"] <= 256:
        raise ConfigurationError("max_runs must be 1..256")
    base_path = reference_path(root, sweep["experiment"])
    base_doc = load_document(base_path)
    base = expand_experiment(base_path, root)
    names = set()
    for axis in sweep["axes"]:
        fields(axis, ("option", "values", "factor"), where="axis")
        name = axis["option"]
        if name in names or name in {change["option"] for change in base["changes"]}:
            raise ConfigurationError(f"ambiguous sweep axis: {name}")
        names.add(name)
        text_value(axis["factor"], "axis factor")
        if not isinstance(axis["values"], list) or not axis["values"]:
            raise ConfigurationError("axis values must be a nonempty list")
        values = [decode_options({name: value})[name] for value in axis["values"]]
        if len({json.dumps(value, sort_keys=True) for value in values}) != len(values):
            raise ConfigurationError("duplicate sweep values")
    count = math.prod(len(axis["values"]) for axis in sweep["axes"])
    if count > sweep["max_runs"]:
        raise ConfigurationError(f"sweep has {count} runs, exceeding max_runs={sweep['max_runs']}")
    from .options import UNITS, VECTOR_UNITS

    expanded = []
    for index, values in enumerate(itertools.product(*(axis["values"] for axis in sweep["axes"]))):
        document = {**base_doc, "id": f"{sweep['id']}-{index + 1:03d}", "changes": list(base_doc["changes"])}
        for axis, value in zip(sweep["axes"], values, strict=True):
            name = axis["option"]
            before = base["runtime_options"][name]
            if decode_options({name: value})[name] == before:
                continue
            unit = UNITS.get(name, VECTOR_UNITS.get(name))
            document["changes"].append({"option": name, "before": {"value": before, "unit": unit} if unit else before,
                                        "after": value, "factor": axis["factor"], "reason": sweep["purpose"]})
        document["variant"] = "baseline" if not document["changes"] else "tuning" if len(document["changes"]) == 1 else "combination"
        resolved = expand_experiment(base_path, root, document=document)
        resolved["sweep_id"] = sweep["id"]
        expanded.append({"configuration": resolved, "resource_estimate": resource_estimate(resolved)})
    return {"schema_version": 1, "sweep_id": sweep["id"], "dry_run": True, "run_count": count, "runs": expanded}
