"""An isolated, bounded process that uses the ordinary production FD runtime."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from .options import argv_for
from .runs import canonical_hash, file_hash, source_identity, write_json

_job_handle = None


def limit_resources(memory_mib):
    """Apply process affinity and address/commit memory limits before JAX loads."""
    memory_bytes = memory_mib * 1024 * 1024
    if os.name == "nt":
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", wintypes.DWORD), ("min_working", ctypes.c_size_t),
                        ("max_working", ctypes.c_size_t), ("active_processes", wintypes.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]

        class IOCounts(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io", IOCounts), ("process_memory", ctypes.c_size_t),
                        ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        process = kernel.GetCurrentProcess()
        allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(process, ctypes.byref(allowed), ctypes.byref(system)):
            raise OSError(ctypes.get_last_error(), "cannot read CPU affinity")
        affinity = allowed.value & -allowed.value
        limits = ExtendedLimits()
        limits.basic.flags = 0x10 | 0x100 | 0x200 | 0x2000
        limits.basic.affinity = affinity
        limits.process_memory = memory_bytes
        limits.job_memory = memory_bytes
        job = kernel.CreateJobObjectW(None, None)
        if not job or not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)) or not kernel.AssignProcessToJobObject(job, process):
            raise OSError(ctypes.get_last_error(), "cannot enforce 1 CPU / memory job limits")
        global _job_handle
        _job_handle = job
    else:
        import resource

        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("this platform cannot enforce a one-CPU managed job")
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))


def synthetic_services(expanded, directory, grid_receipt):
    """Construct documented small inputs; every step still uses make_solver_global."""
    import numpy as np

    from zhenmode.model.runtime.run import default_services
    from zhenmode.model.solver.geometry.grid import GlobalOceanGrid

    case = expanded["case"]
    spec = case["grid"]
    nx, ny = spec["nx"], spec["ny"]
    z = np.asarray(spec["z_levels"]["value"], dtype=float)
    nz = len(z)
    lat_max = case["domain"]["lat_max"]["value"]
    lat = np.linspace(-lat_max, lat_max, ny)
    lon = (np.arange(nx) + 0.5) * 360.0 / nx
    cos_lat = np.cos(np.deg2rad(lat))
    radius = 6.371e6
    dx = np.broadcast_to(radius * cos_lat * np.deg2rad(360.0 / nx), (nx, ny)).copy()
    coriolis = np.zeros((nx, ny)) if case["domain"]["coriolis"] == "none" else np.broadcast_to(2 * 7.2921e-5 * np.sin(np.deg2rad(lat)), (nx, ny)).copy()
    grid = GlobalOceanGrid(lon=lon, lat=lat, dx_2d=dx, dy=float(radius * np.deg2rad(lat[1] - lat[0])),
                           cos_lat=cos_lat, f=coriolis, z=z, dz=-np.diff(z), nz=nz,
                           depth=np.full((nx, ny), spec["depth"]["value"]), wet_mask=np.ones((nx, ny)),
                           ocean_mask=np.ones((nx, ny), dtype=bool), land_mask=np.zeros((nx, ny)),
                           wet_mask_3d=np.ones((nx, ny, nz)), nx=nx, ny=ny)
    grid_receipt["grid"] = grid
    temperature = np.full((nx, ny, nz), case["initial"]["temperature"]["value"], dtype=float)
    salinity = np.full_like(temperature, case["initial"]["salinity"]["value"])
    increment = case["forcing"]["monthly_tau_increment"]["value"]
    monthly = [(np.full((nx, ny), (month + 1) * increment), np.zeros((nx, ny))) for month in range(12)]
    input_path = directory / "synthetic-inputs.npz"
    with input_path.open("xb") as stream:
        np.savez_compressed(stream, T_init=temperature, S_init=salinity, wind=np.asarray(monthly), lon=lon, lat=lat, z=z,
                            depth=grid.depth, f=coriolis, dx_2d=dx, dy=grid.dy)
    return replace(default_services(),
                       make_global_grid=lambda *args, **kwargs: grid,
                       get_initial_fields=lambda grid: (temperature, salinity),
                       build_seasonal_wind_global=lambda grid, year: monthly,
                       input_files=lambda *args, **kwargs: {"synthetic_case": input_path},
                       source_identity=lambda: {"schema_version": 2, "all_package_files": source_identity()},
                       resolve_grid_dimensions=lambda *args, **kwargs: (nx, ny))


def bind_external_inputs(manifest):
    import zhenmode.model.inputs.forcing.reanalysis as reanalysis
    import zhenmode.model.inputs.initial_conditions as climatology
    from zhenmode.model.config import DEFAULT_CONFIG

    data = manifest["data"]
    bathymetry = data["bathymetry"]["resolved_path"]
    config = replace(DEFAULT_CONFIG, bathymetry_file=bathymetry.removesuffix(".npz"))
    climatology.WOA_FILES = {name: data[name]["resolved_path"].removesuffix(".npz") for name in ("temperature", "salinity")}
    wind_paths = [Path(value["resolved_path"]) for key, value in data.items() if key.startswith("wind-")]
    if len({path.parent for path in wind_paths}) != 1:
        raise ValueError("all selected wind caches must share one cache directory")
    reanalysis.WIND_CACHE_DIR = str(wind_paths[0].parent)
    if "air" in data:
        reanalysis.AIR_CACHE_DIR = str(Path(data["air"]["resolved_path"]).parent)
    return config


def verify_selected_inputs(selected, manifest, options):
    """Check loader roles and frozen content, not just an unordered path set."""
    expected = {}
    for role, record in manifest["data"].items():
        if role.startswith("wind-"):
            month = int(role[5:]) - 1 if role != "wind-fixed" else int(options["month"][5:7]) - 1
            year = options["wind_year"] if role != "wind-fixed" else int(options["month"][:4])
            loader_role = f"wind_{(year - 1948) * 12 + month}"
        else:
            loader_role = role
        expected[loader_role] = record
    if set(selected) != set(expected):
        raise ValueError("selected loader roles do not match frozen case roles")
    for role, path in selected.items():
        path = Path(path).resolve()
        frozen = expected[role]
        if str(path) != frozen["resolved_path"]:
            raise ValueError(f"selected loader path differs for role {role}")
        if file_hash(path) != frozen["observed_sha256"]:
            raise ValueError(f"input changed after run freeze: {role} ({path})")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    directory = args.run_directory.resolve()
    expanded = json.loads((directory / "expanded.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    limit_resources(expanded["resources"]["memory_mib"])
    if source_identity() != manifest["source_identity"]:
        raise RuntimeError("actual installed execution files changed after the run was frozen")
    import jax
    import numpy as np

    import zhenmode.model.runtime.run as application
    from zhenmode.model.runtime.run import default_services, run_main
    from zhenmode.provenance.sources import source_root

    if jax.default_backend() != "cpu":
        raise RuntimeError("managed local execution requires CPU")
    options = expanded["runtime_options"] | {"tag": "run", "out_dir": str(directory / "model"), "log_dir": str(directory / "logs")}
    grid_receipt = {}
    synthetic = expanded["case"]["grid"]["kind"] == "synthetic"
    if synthetic:
        services = synthetic_services(expanded, directory, grid_receipt)
    else:
        services = default_services(bind_external_inputs(manifest))
        build_grid = services.make_global_grid
        def capture_grid(*args, **kwargs):
            grid = build_grid(*args, **kwargs)
            grid_receipt["grid"] = grid
            return grid
        services = replace(services, make_global_grid=capture_grid)
        # Validate the actual loader selections, including .npz precedence.
        from zhenmode.model.runtime.cli import parse_run_configuration

        config = parse_run_configuration(argv_for(options))
        verify_selected_inputs(services.input_files(config.args), manifest, options)
    sys.argv = ["zhenmode-production-worker", *argv_for(options)]
    previous_stdout = sys.stdout
    try:
        code = run_main(services, source_root(application.__file__))
    finally:
        if sys.stdout is not previous_stdout:
            sys.stdout.file.close()
            sys.stdout = previous_stdout
    result_path = directory / "model" / "global_run.npz"
    loaded = {}
    package_root = source_root(application.__file__) / "zhenmode"
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("zhenmode") and path and Path(path).suffix == ".py":
            path = Path(path).resolve()
            if path.is_relative_to(package_root):
                loaded[path.relative_to(package_root).as_posix()] = file_hash(path)
    loaded[Path(__file__).resolve().relative_to(package_root).as_posix()] = file_hash(__file__)
    for name, sha in loaded.items():
        if manifest["source_identity"].get(name) != sha:
            raise RuntimeError(f"executed source file changed during execution: {name}")
    from zhenmode.model.io.restart import fingerprint

    grid = grid_receipt["grid"]
    physical_grid_path = directory / "actual-physical-grid.npz"
    with physical_grid_path.open("xb") as stream:
        np.savez_compressed(stream, **{name: getattr(grid, name) for name in ("lon", "lat", "z", "depth", "wet_mask", "ocean_mask", "dx_2d", "dy")})
    report = {"executed_source_files": loaded, "result_path": str(result_path),
              "execution_hardware": {"backend": jax.default_backend(), "device_kind": jax.devices()[0].device_kind},
              "resource_enforcement": {"cpu_affinity": 1, "memory_mib": expanded["resources"]["memory_mib"]},
              "physical_grid_sha256": fingerprint({name: getattr(grid, name) for name in ("lon", "lat", "z", "depth", "wet_mask", "dx_2d", "dy")}),
              "physical_grid_path": str(physical_grid_path), "physical_grid_artifact_sha256": file_hash(physical_grid_path),
              "execution_grid_metadata": {"nx": grid.nx, "ny": grid.ny, "nz": grid.nz,
                                           "longitude_spacing_deg": float(grid.lon[1] - grid.lon[0]),
                                           "latitude_spacing_deg": float(grid.lat[1] - grid.lat[0]), "isotropic_angular_spacing": bool(np.allclose(np.diff(grid.lon), np.diff(grid.lat)[0]))}}
    with np.load(result_path, allow_pickle=False) as result:
        for name in ("verdict", "requested_steps", "accepted_steps", "duration_complete"):
            report[name] = result[name].item()
        report["forcing_provenance"] = json.loads(str(result["forcing_provenance_json"]))
        report["result_source_identity"] = json.loads(str(result["source_identity_json"]))
    selected_inputs = report["forcing_provenance"]["selected_files"]
    if not synthetic:
        verify_selected_inputs({name: value["path"] for name, value in selected_inputs.items()}, manifest, options)
        for name, value in selected_inputs.items():
            if file_hash(value["path"]) != value["sha256"]:
                raise ValueError(f"effective input identity changed during execution: {name}")
    report["data_sha256"] = canonical_hash({name: value["sha256"] for name, value in selected_inputs.items()})
    report["result_sha256"] = file_hash(result_path)
    write_json(directory / "worker-report.json", report, create=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
