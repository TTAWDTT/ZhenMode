"""Explicit fixed-partial CT/SR input bridge for the actual FD factory.

This prepares data, effective spherical-area metrics and declared reference
pressure. It does not integrate, select complete mechanisms or qualify a case.
"""

from __future__ import annotations

import json
from pathlib import Path

import netCDF4
import numpy as np

from zhenmode.execution.initialization import _checked_file
from zhenmode.model.config import G_EARTH, OMEGA, R_EARTH, RHO_0
from zhenmode.model.solver.geometry.grid import GlobalOceanGrid, fixed_reference_nodal_cells
from zhenmode.model.solver.physics.teos10 import validate_state
from zhenmode.provenance.sources import (
    load_json,
    production_source_modules,
    sha256_file,
    source_paths,
    source_root,
)

_FIELDS = {
    "lon",
    "lat",
    "lon_bounds",
    "lat_bounds",
    "depth",
    "wet_mask",
    "node_depth_m",
    "thickness_m",
    "wet_node_mask",
    "area",
    "dx_2d",
    "dy",
    "cos_lat",
    "f",
    "ct",
    "sr",
    "eos_pressure_dbar",
}

# Rebuilding trig metrics with another NumPy/libm may differ by a few binary64
# ULPs. Keep exact shapes and zero values; permit only an operation-level
# relative rounding allowance, far below any physical geometry tolerance.
_REBUILT_METRIC_RTOL = 16 * np.finfo(np.float64).eps


def _policy(policy):
    if (
        set(policy)
        != {
            "schema_version",
            "horizontal_metric",
            "pressure_definition",
            "inactive_ct_deg_c",
            "inactive_sr_g_kg",
        }
        or type(policy["schema_version"]) is not int
        or policy["schema_version"] != 1
        or policy["horizontal_metric"] != "geographic_area_v1"
        or policy["pressure_definition"] != "fixed_boussinesq_reference_v1"
        or any(
            type(policy[k]) not in (int, float) for k in ("inactive_ct_deg_c", "inactive_sr_g_kg")
        )
    ):
        raise ValueError(
            "FD input policy requires explicit area, pressure and inactive CT/SR definitions"
        )
    validate_state(policy["inactive_sr_g_kg"], policy["inactive_ct_deg_c"], 0)


def _metrics_and_geometry(g):
    lon, lat = g["lon"], g["lat"]
    nx, ny = len(lon), len(lat)
    if (
        nx < 2
        or ny < 2
        or not np.allclose(lon, (np.arange(nx) + 0.5) * 360 / nx, rtol=0, atol=1e-10)
        or not np.allclose(lat, (np.arange(ny) + 0.5) * 180 / ny - 90, rtol=0, atol=1e-10)
        or not np.allclose(
            g["lon_bounds"],
            np.column_stack((np.arange(nx), np.arange(1, nx + 1))) * 360 / nx,
            rtol=0,
            atol=1e-10,
        )
        or not np.allclose(
            g["lat_bounds"],
            np.column_stack((np.arange(ny), np.arange(1, ny + 1))) * 180 / ny - 90,
            rtol=0,
            atol=1e-10,
        )
    ):
        raise ValueError("FD input bridge requires regular full-global geographic bounds")
    cells = fixed_reference_nodal_cells(-g["node_depth_m"], g["depth"], g["wet_mask"])
    for name in ("thickness_m", "wet_node_mask"):
        if not np.array_equal(cells[name], g[name]):
            raise ValueError("FD input reference capacities/mask are inconsistent")
    area = (
        R_EARTH**2
        * np.deg2rad(np.diff(g["lon_bounds"], axis=1))
        * np.diff(np.sin(np.deg2rad(g["lat_bounds"])), axis=1).T
    )
    if (
        g["area"].shape != (nx, ny)
        or not np.isfinite(g["area"]).all()
        or not np.allclose(g["area"], area, rtol=1e-10, atol=1e-6)
    ):
        raise ValueError("FD input area disagrees with spherical bounds")
    dy = R_EARTH * np.deg2rad(180 / ny)
    dx = g["area"] / dy
    cosine = np.cos(np.deg2rad(lat))
    length = dx / cosine[None, :]
    if not np.allclose(length, length[0, 0], rtol=1e-12, atol=0):
        raise ValueError("FD input area does not meet compatible dx=L*cos_lat metric rule")
    pressure = np.broadcast_to(
        (RHO_0 * G_EARTH * g["node_depth_m"] * 1e-4)[None, None, :], g["wet_node_mask"].shape
    )
    return {
        "dy": np.asarray(dy),
        "dx_2d": dx,
        "cos_lat": cosine,
        "f": np.broadcast_to((2 * OMEGA * np.sin(np.deg2rad(lat)))[None, :], (nx, ny)).copy(),
        "eos_pressure_dbar": pressure.copy(),
    }


def prepare_fd_native_inputs(native_prepared, policy_file, output):
    """Export full finite FD inputs; numerical sentinels fill inactive nodes only."""
    native_prepared, policy_file, output = map(Path, (native_prepared, policy_file, output))
    receipt = native_prepared / "native_initialization.json"
    receipt_sha, policy_sha = sha256_file(receipt), sha256_file(policy_file)
    parent, policy = load_json(receipt), load_json(policy_file)
    _policy(policy)
    if (
        parent.get("geometry") != "fixed_partial_v1"
        or parent.get("complete_wet_support") is not True
        or parent.get("status")
        not in {"native_points_prepared", "native_points_completed_with_bottom_assumptions"}
    ):
        raise ValueError(
            "FD input bridge requires complete, explicit fixed-partial native point data"
        )
    source = _checked_file(native_prepared, parent["output"])
    geometry = native_prepared / parent["geometry_output"]["path"]
    identities = {
        receipt: receipt_sha,
        policy_file: policy_sha,
        source: parent["output"]["sha256"],
        geometry: parent["geometry_output"]["sha256"],
    }
    for path, expected in identities.items():
        if sha256_file(path) != expected:
            raise ValueError("FD native input identity mismatch")
    with np.load(geometry, allow_pickle=False) as data:
        g = {
            k: data[k].copy()
            for k in (
                "lon",
                "lat",
                "lon_bounds",
                "lat_bounds",
                "depth",
                "wet_mask",
                "node_depth_m",
                "thickness_m",
                "wet_node_mask",
                "area",
            )
        }
    metrics = _metrics_and_geometry(g)
    with netCDF4.Dataset(source) as data:
        for name, expected, units in (
            ("lon", g["lon"], "degrees_east"),
            ("lat", g["lat"], "degrees_north"),
            ("depth", g["node_depth_m"], "meters"),
        ):
            if data[name].units != units or not np.array_equal(data[name][:], expected):
                raise ValueError("FD source coordinates/units disagree with native geometry")
        fields = {}
        for name, units in (("ct", "degrees_celsius"), ("sr", "g kg-1")):
            if (
                data[name].dimensions != ("time", "depth", "lat", "lon")
                or data[name].units != units
            ):
                raise ValueError("FD source CT/SR dimensions or units disagree")
            fields[name] = (
                np.ma.asarray(data[name][0], dtype=float).filled(np.nan).transpose(2, 1, 0)
            )
        time = data["time"]
        values = np.ma.asarray(time[:])
        if (
            time.dimensions != ("time",)
            or values.shape != (1,)
            or np.ma.is_masked(values)
            or not np.isfinite(values).all()
            or getattr(time, "calendar", None) not in {"gregorian", "proleptic_gregorian"}
        ):
            raise ValueError("FD native initial time requires one finite Gregorian record")
        time_definition = {
            "values": np.asarray(values).tolist(),
            "units": time.units,
            "calendar": time.calendar,
        }
    wet = g["wet_node_mask"].astype(bool)
    if any(
        values.shape != wet.shape or not np.isfinite(values[wet]).all()
        for values in fields.values()
    ):
        raise ValueError("FD native data still contain missing/nonfinite wet CT/SR")
    ct = np.where(wet, fields["ct"], policy["inactive_ct_deg_c"])
    sr = np.where(wet, fields["sr"], policy["inactive_sr_g_kg"])
    # Validate every evaluated pressure, including inactive reference nodes.
    # No pressure clipping, node deletion or wet infill occurs in this bridge.
    validate_state(sr, ct, metrics["eos_pressure_dbar"])
    output.mkdir(parents=True, exist_ok=False)
    software = {
        n: sha256_file(p)
        for n, p in source_paths(source_root(__file__), production_source_modules()).items()
    }
    report = {
        "schema_version": 1,
        "status": "running",
        "data_kind": parent["data_kind"],
        "parent_native_receipt_sha256": receipt_sha,
        "input_sha256": {str(p): s for p, s in identities.items()},
        "policy": policy,
        "package_source_sha256": software,
        "geometry": "fixed_partial_v1",
        "thermodynamics": "teos10_reference",
        "temperature_definition": "CT degrees Celsius",
        "salinity_definition": "SR g/kg approximates SA; no geographic salinity anomaly",
        "initial_time_definition": time_definition,
        "source_time_definition": parent["source_time_definition"],
        "pressure_definition": {
            "id": policy["pressure_definition"],
            "rho0_kg_m3": RHO_0,
            "g_m_s2": G_EARTH,
            "units": "dbar",
            "formula": "rho0*g*depth_m/10000",
            "exact_geographic_pressure": False,
        },
        "source_geography_certified": False,
        "initialization_entry_verified": False,
        "execution_ready": False,
        "climate_qualification": False,
        "completed_timesteps": 0,
        "inactive_value_scope": "numerical_sentinels_only; no wet values are replaced",
        "horizontal_metric_scope": "effective dx=A/dy, dy=R*dlat; compatible contact metrics; physical/dynamical accuracy remains separate",
    }
    try:
        arrays = {**g, **metrics, "ct": ct, "sr": sr}
        target = output / "fd-native-inputs.npz"
        np.savez_compressed(target, **arrays)
        volume = g["area"][..., None] * g["thickness_m"]
        midpoint_dx = R_EARTH * np.deg2rad(360 / len(g["lon"])) * metrics["cos_lat"][None, :]
        report.update(
            status="fd_native_inputs_prepared",
            shape=list(wet.shape),
            wet_nodes=int(wet.sum()),
            pressure_max_dbar=float(metrics["eos_pressure_dbar"].max()),
            wet_pressure_max_dbar=float(metrics["eos_pressure_dbar"][wet].max()),
            reference_volume_m3=float(volume.sum()),
            max_dx_relative_change_from_midpoint=float(
                np.max(np.abs(metrics["dx_2d"] / midpoint_dx - 1))
            ),
            maximum_area_multiplication_abs_error_m2=float(
                np.max(np.abs(metrics["dx_2d"] * metrics["dy"] - g["area"]))
            ),
            output={
                "path": target.name,
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            },
        )
        for path, expected in identities.items():
            if sha256_file(path) != expected:
                raise ValueError("FD native input source/policy changed during preparation")
        if software != {
            n: sha256_file(p)
            for n, p in source_paths(source_root(__file__), production_source_modules()).items()
        }:
            raise ValueError("executed FD native input package changed")
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", reason_type=type(error).__name__, reason=str(error))
        raise
    finally:
        (output / "fd_initialization.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
    return report


def load_fd_native_inputs(directory):
    """Read the prepared bundle for explicit make_solver_global/init_state calls.

    Returns grid, CT, SR and full reference pressure. A consumer must select
    fixed_partial_v1/teos10_reference and its actual mechanism configuration.
    This loader performs no integration or implicit scheme/config switching.
    """
    directory = Path(directory)
    report = load_json(directory / "fd_initialization.json")
    if (
        report.get("schema_version") != 1
        or report.get("status") != "fd_native_inputs_prepared"
        or report.get("geometry") != "fixed_partial_v1"
        or report.get("thermodynamics") != "teos10_reference"
    ):
        raise ValueError("completed FD native input receipt is required")
    _policy(report["policy"])
    source = _checked_file(directory, report["output"])
    with np.load(source, allow_pickle=False) as data:
        if set(data.files) != _FIELDS:
            raise ValueError("FD native input bundle fields disagree")
        arrays = {k: data[k].copy() for k in data.files}
    derived = _metrics_and_geometry(arrays)
    for name, values in derived.items():
        if (
            arrays[name].shape != values.shape
            or arrays[name].dtype.kind not in "fiu"
            or not np.allclose(values, arrays[name], rtol=_REBUILT_METRIC_RTOL, atol=0)
        ):
            raise ValueError(
                "FD native input metrics/reference pressure disagree with their definitions"
            )
    if any(arrays[name].shape != arrays["wet_node_mask"].shape for name in ("ct", "sr")):
        raise ValueError("FD native CT/SR must have the full grid shape")
    inactive = ~arrays["wet_node_mask"].astype(bool)
    if not np.all(arrays["ct"][inactive] == report["policy"]["inactive_ct_deg_c"]) or not np.all(
        arrays["sr"][inactive] == report["policy"]["inactive_sr_g_kg"]
    ):
        raise ValueError("FD inactive values disagree with the declared numerical sentinels")
    validate_state(arrays["sr"], arrays["ct"], arrays["eos_pressure_dbar"])
    depth = arrays["node_depth_m"]
    ocean = arrays["wet_mask"] > 0
    grid = GlobalOceanGrid(
        lon=arrays["lon"],
        lat=arrays["lat"],
        dx_2d=arrays["dx_2d"],
        dy=float(arrays["dy"]),
        cos_lat=arrays["cos_lat"],
        f=arrays["f"],
        z=-depth,
        dz=np.diff(depth),
        nz=len(depth),
        depth=arrays["depth"],
        wet_mask=arrays["wet_mask"],
        ocean_mask=ocean,
        land_mask=~ocean,
        wet_mask_3d=arrays["wet_node_mask"].astype(float),
        nx=len(arrays["lon"]),
        ny=len(arrays["lat"]),
    )
    if sha256_file(source) != report["output"]["sha256"]:
        raise ValueError("FD native input changed while loading")
    return grid, arrays["ct"], arrays["sr"], arrays["eos_pressure_dbar"]
