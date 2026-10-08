"""Real native FD wind-driven integration, bounded CUDA execution and restarts.

This deliberately named component reads all JRA fields but applies wind stress
only. Heat, freshwater, ice, GM and Redi are not silently counted as enabled.
It reuses the actual core, its stage budgets and the strict restart format.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from zhenmode.provenance.sources import (
    load_json,
    production_source_modules,
    sha256_file,
    source_paths,
    source_root,
)


def _write(path, value):
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_timestep(params):
    """Reject unsafe explicit steps before publishing any accepted state.

    Gershgorin bounds apply to the actual fixed-partial contact Laplacian
    and its square. The fast-wave bound includes both horizontal directions.
    This is a conservative startup bound, not a nonlinear stability proof.
    """
    import numpy as np

    from zhenmode.model.config import G_EARTH
    from zhenmode.model.solver.numerics.backend import jnp
    from zhenmode.model.solver.numerics.contacts import incoming

    wet = np.asarray(params.wet_mask_z) > .5
    volume = params.dx_2d[..., None] * params.dy * params.dz_node
    fx = params.face_contacts[0] * params.dy * params.inv_dx[None]
    cosine = .5 * (params.cos_lat + jnp.roll(params.cos_lat, -1))
    fy = (params.face_contacts[1] * cosine[None, None, :, None]
          * (params.dx_2d / params.cos_lat[None, :])[None, ..., None] * params.inv_dy)
    diagonal = np.asarray((fx.sum(0) + incoming(fx, 0)
                           + fy.sum(0) + incoming(fy, 1)) / volume)
    lambda_h = 2 * float(diagonal[wet].max())
    widths = np.asarray(params.dz_node)
    spacing = np.diff(np.asarray(params.node_depth_m), axis=-1)
    interface = (wet[..., :-1] & wet[..., 1:]) / spacing
    vertical_rate = (np.pad(interface, ((0, 0), (0, 0), (1, 0)))
                     + np.pad(interface, ((0, 0), (0, 0), (0, 1)))) / widths
    lambda_v = 2 * float(vertical_rate[wet].max())
    rate = (max(params.nu_h, params.kappa_h) * lambda_h
            + max(params.nu_bi, params.kappa_bi) * lambda_h**2
            + max(params.nu_v, params.kappa_v) * lambda_v)
    linear_limit = 2 / rate if rate > 0 else float("inf")
    depth = float(np.asarray(params.H_sw)[np.asarray(params.wet_mask) > .5].max())
    metric = min(float(np.asarray(params.dx_2d).min()), float(params.dy))
    wave_limit = .5 * metric / np.sqrt(2 * G_EARTH * depth)
    limit = min(linear_limit, wave_limit)
    if params.dt > limit:
        raise ValueError(f"requested dt {params.dt:g}s exceeds conservative explicit limit {limit:g}s")
    return {"linear_limit_s": None if not np.isfinite(linear_limit) else linear_limit,
            "external_wave_limit_s": float(wave_limit), "maximum_dt_s": float(limit)}


def run_fd_wind(
    native_prepared,
    forcing_manifest,
    output,
    *,
    start,
    dt_seconds,
    steps,
    wall_seconds=360,
    resume=None,
    polar_cap_rows=2,
    polar_cap_taper=3,
    match_transport=False,
):
    """Launch one CUDA worker; retain accepted state and failure/resource records."""
    if os.name != "posix":
        raise ValueError("native GPU integration requires Linux/WSL; invoke this command there")
    if type(steps) is not int or steps <= 0 or not 0 < float(dt_seconds) < float("inf"):
        raise ValueError("positive finite dt and positive integer steps are required")
    if type(wall_seconds) is not int or not 20 <= wall_seconds <= 10800:
        raise ValueError("CUDA wall budget must be 20..10800 seconds")
    for value in (polar_cap_rows, polar_cap_taper):
        if type(value) is not int or value < 0:
            raise ValueError("polar cap rows/taper must be nonnegative integers")
    timestamp = datetime.fromisoformat(start)
    if timestamp.tzinfo is not None:
        raise ValueError("start must be a timezone-free Gregorian ISO timestamp")
    output = Path(output).resolve()
    config = {
        "native_prepared": str(Path(native_prepared).resolve()),
        "forcing_manifest": str(Path(forcing_manifest).resolve()),
        "output": str(output),
        "start": start,
        "dt_seconds": float(dt_seconds),
        "steps": steps,
        "resume": None if resume is None else str(Path(resume).resolve()),
        "polar_cap_rows": polar_cap_rows,
        "polar_cap_taper": polar_cap_taper,
        "match_transport": bool(match_transport),
    }
    for name in ("native_prepared", "forcing_manifest"):
        if not Path(config[name]).exists():
            raise FileNotFoundError(config[name])
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "configuration.json"
    _write(config_path, config)
    from zhenmode.execution.resources import run_cuda_worker

    env = dict(
        os.environ,
        JAX_PLATFORMS="cuda",
        CUDA_VISIBLE_DEVICES="0",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION=".40",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    resources = {
        "cpu": 1,
        "wall_seconds": wall_seconds,
        "memory_mib": 8192,
        "termination_grace_seconds": 10,
    }
    with (output / "worker.log").open("x", encoding="utf8") as log:
        result, resource_report = run_cuda_worker(
            [sys.executable, "-m", __name__, "--worker-config", str(config_path)],
            cwd=output,
            env=env,
            stdout=log,
            resources=resources,
        )
    _write(output / "resources.json", resource_report | {"returncode": result.returncode})
    path = output / "run.json"
    report = load_json(path) if path.exists() else {"completed_timesteps": 0}
    if result.returncode != 0:
        report.update(
            status="failed",
            stop_reason=resource_report["stop_reason"],
            returncode=result.returncode,
        )
    report["resources"] = resource_report
    _write(path, report)
    return report


def integrate_fd_wind(config, *, required_backend="gpu"):
    """Use installed/check-out public inputs and core; never synthesize ocean data."""
    import signal

    import jax
    import numpy as np

    from zhenmode.execution.runs import environment_identity
    from zhenmode.model.config import PhysicsConfig
    from zhenmode.model.diagnostics.budgets import (
        METRIC_NAMES,
        NONLINEAR_PROCESS_NAMES,
        SOURCE_NAMES,
        STAGE_NAMES,
        accumulate_budget,
        empty_budget,
        make_budget_step,
    )
    from zhenmode.model.inputs.forcing.jra55 import JRA55Forcing
    from zhenmode.model.io.restart import load_restart, make_restart_contract, save_restart
    from zhenmode.model.solver.factory import make_solver_global
    from zhenmode.model.solver.physics.air_sea import open_water_fluxes
    from zhenmode.model.solver.physics.teos10 import (
        potential_from_conservative,
        surface_freezing_ct,
        validate_state,
    )
    from zhenmode.model.solver.state import JaxStateG
    from zhenmode.preparation.fd import load_fd_native_inputs

    output = Path(config["output"])
    prepared = Path(config["native_prepared"])
    forcing = Path(config["forcing_manifest"])
    source_files = source_paths(source_root(__file__), production_source_modules())
    sources = {n: sha256_file(p) for n, p in source_files.items()}
    report = {
        "status": "running",
        "profile": "real_JRA_wind_dynamics_component_v1",
        "completed_timesteps": 0,
        "requested_additional_steps": config["steps"],
        "full_case_qualification": False,
        "sources": sources,
        "dt_seconds": config["dt_seconds"],
        "unused_surface_processes": [
            "heat",
            "rain",
            "snow",
            "evaporation",
            "runoff",
            "calving",
            "salt_restoring",
            "ice",
        ],
        "freshwater_routing_applied": False,
        "polar_filter_scope": "fixed blend each step; not a physical mixing source",
    }
    step = 0
    state = None
    contract = None
    totals = empty_budget()
    history = {}
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop)

    def checkpoint():
        save_restart(
            output / "checkpoint.npz",
            state._asdict(),
            contract,
            step=step,
            counters={"accepted_steps": step},
            cumulative=totals,
            history=history,
        )

    def sync(tree):
        jax.tree.map(
            lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a, tree
        )

    def progress():
        report["completed_timesteps"] = step
        report["elapsed_simulation_seconds"] = step * config["dt_seconds"]
        _write(output / "run.json", report)

    try:
        report["backend"] = jax.default_backend()
        report["device"] = jax.devices()[0].device_kind
        progress()
        if report["backend"] != required_backend or len(jax.devices()) != 1:
            raise ValueError("requested single-device integration backend unavailable")
        grid, ct, sr, pressure = load_fd_native_inputs(prepared)
        physics = PhysicsConfig(thermodynamics="teos10_reference")
        _, initialize, _, params, _, _ = make_solver_global(
            grid,
            physics,
            dt=config["dt_seconds"],
            dt_bt=config["dt_seconds"],
            mode_split=True,
            return_params=True,
            dynamic_forcing=True,
            column_geometry="fixed_partial_v1",
            conservative_kv=True,
            localize_conv=True,
            eos_pressure_dbar=pressure,
            polar_cap_rows=config["polar_cap_rows"],
            polar_cap_taper=config["polar_cap_taper"],
            match_barotropic_transport=config["match_transport"],
        )
        report["time_step_limits"] = _validate_timestep(params)
        origin = (datetime.fromisoformat(config["start"]) - datetime(1970, 1, 1)).total_seconds()
        input_ids = {
            "native_receipt": sha256_file(prepared / "fd_initialization.json"),
            "native_arrays": sha256_file(prepared / "fd-native-inputs.npz"),
            "forcing_manifest": sha256_file(forcing),
        }
        contract = make_restart_contract(
            grid,
            params,
            dtype="float64",
            forcing=input_ids | {"start": config["start"], "coupling": "wind_only_LY2009_Gill"},
            controls={"profile": report["profile"], "state_save": "accepted_only"},
            code_paths=source_files,
            execution=environment_identity()
            | {"backend": report["backend"], "device": report["device"]},
        )
        if config["resume"]:
            restored = load_restart(config["resume"], contract)
            state = JaxStateG(*(jax.numpy.asarray(restored.state[n]) for n in JaxStateG._fields))
            step = restored.step
            totals = jax.tree.map(jax.numpy.asarray, restored.cumulative)
            history = {n: np.asarray(v) for n, v in restored.history.items()}
        else:
            state = initialize(T_init=ct, S_init=sr)
        sync(state)
        report.update(
            shape=[grid.nx, grid.ny, grid.nz],
            data_kind=load_json(prepared / "fd_initialization.json")["data_kind"],
            physics=asdict(physics),
            initial_step=step,
            input_sha256=input_ids,
            solver_options={
                k: config[k] for k in ("polar_cap_rows", "polar_cap_taper", "match_transport")
            },
        )
        reader = JRA55Forcing(
            forcing,
            lon=grid.lon,
            lat=grid.lat,
            wet_mask=grid.wet_mask,
            start_seconds=origin + step * params.dt,
            end_seconds=origin + (step + config["steps"]) * params.dt,
        )
        report["initialization_data_kind"] = report["data_kind"]
        report["forcing_data_kind"] = reader.data_kind
        if report["data_kind"] != reader.data_kind:
            report["data_kind"] = "mixed"
        wet = np.asarray(params.wet_mask) > 0
        advance = make_budget_step(params)

        @jax.jit
        def stress(current, air):
            air = air._replace(
                temperature_k=jax.numpy.where(wet, air.temperature_k, 273.15),
                pressure_pa=jax.numpy.where(wet, air.pressure_pa, 101325.0),
            )
            pt = potential_from_conservative(current.S[:, :, 0], current.T[:, :, 0])
            flux = open_water_fluxes(air, pt, current.u[:, :, 0], current.v[:, :, 0])
            return (
                jax.numpy.where(wet, flux.tau_x, 0),
                jax.numpy.where(wet, flux.tau_y, 0),
                jax.numpy.zeros(wet.shape),
            )

        checkpoint()
        progress()
        for _ in range(config["steps"]):
            if stopping:
                raise InterruptedError("resource supervisor requested a saved stop")
            timing = {}
            tick = time.monotonic()
            air = reader.sample(
                origin + step * params.dt, interval_end_seconds=origin + (step + 1) * params.dt
            )
            timing["forcing_read_s"] = time.monotonic() - tick
            freeze = np.asarray(surface_freezing_ct(state.S[:, :, 0]))
            report["full_surface_blockers"] = {
                "at_or_below_freezing": int(np.sum((np.asarray(state.T[:, :, 0]) <= freeze) & wet)),
                "nonzero_snow": int(np.sum((np.asarray(air.snow) != 0) & wet)),
                "nonzero_calving": int(np.sum((np.asarray(air.calving) != 0) & wet)),
            }
            tick = time.monotonic()
            wind = stress(state, air)
            sync(wind)
            timing["bulk_stress_s"] = time.monotonic() - tick
            tick = time.monotonic()
            updated, budget = advance(state, forcing=wind)
            sync((updated, budget))
            timing["core_and_budget_s"] = time.monotonic() - tick
            if any(not np.isfinite(np.asarray(v)).all() for v in updated):
                raise ValueError(
                    "nonfinite integrated state rejected; last accepted checkpoint retained"
                )
            validate_state(updated.S, updated.T, params.eos_pressure_dbar)
            state = updated
            step += 1
            totals = accumulate_budget(totals, budget)
            for name, value in budget.items():
                row = np.asarray(value)[None]
                history[name] = row if name not in history else np.concatenate((history[name], row))
            report.setdefault("timings", []).append(timing)
            checkpoint()
            progress()
            print(f"saved accepted GPU step {step}", flush=True)
        if sources != {n: sha256_file(p) for n, p in source_files.items()} or input_ids != {
            "native_receipt": sha256_file(prepared / "fd_initialization.json"),
            "native_arrays": sha256_file(prepared / "fd-native-inputs.npz"),
            "forcing_manifest": sha256_file(forcing),
        }:
            raise ValueError("input/software changed during integration")
        report["status"] = "completed_component"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        if state is not None and contract is not None:
            checkpoint()
        raise
    finally:
        progress()
        _write(
            output / "budgets.json",
            {
                "metrics": METRIC_NAMES,
                "stages": STAGE_NAMES,
                "sources": SOURCE_NAMES,
                "nonlinear_processes": NONLINEAR_PROCESS_NAMES,
                "totals": {k: np.asarray(v).tolist() for k, v in totals.items()},
                "steps": {k: v.tolist() for k, v in history.items()},
            },
        )
    report["checkpoint"] = {
        "path": "checkpoint.npz",
        "sha256": sha256_file(output / "checkpoint.npz"),
    }
    progress()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-config", required=True)
    args = parser.parse_args()
    integrate_fd_wind(load_json(args.worker_config))
