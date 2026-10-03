"""Observe a frozen real-input production run; never change numerical operators.

Use the bounded runner around this command. A plan supplies the identical CLI
and selected data for each revision. --source-root is only for an independently
frozen historical source tree; omit it to observe the installed current wheel.
"""

import argparse
import hashlib
import importlib
import json
import sys
import time
from dataclasses import replace
from functools import partial
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--layout", choices=("legacy", "canonical"), required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--prepare-only", action="store_true",
                        help="export the unchanged production WOA interpolation, without time steps")
    args = parser.parse_args()
    started = time.perf_counter()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    for record in plan["data"].values():
        if digest(record["path"]) != record["sha256"]:
            raise ValueError("frozen input changed: " + record["path"])
    if "restart" in plan and digest(plan["restart"]["path"]) != plan["restart"]["sha256"]:
        raise ValueError("frozen restart changed")
    if args.source_root:
        sys.path.insert(0, str(args.source_root.resolve()))
    import jax
    import numpy as np

    from ocean_solver.runtime import application, entry
    from ocean_solver.runtime.cli import parse_run_configuration

    if jax.default_backend() != "cpu":
        raise RuntimeError("global local probe requires CPU")
    package_root = Path(entry.__file__).resolve().parents[1]
    if args.source_root and package_root.parent != args.source_root.resolve():
        raise RuntimeError("historical execution did not load the frozen source")
    owner = "data" if args.layout == "legacy" else "forcing"
    wind = importlib.import_module(f"ocean_solver.{owner}.wind")
    air = importlib.import_module(f"ocean_solver.{owner}.air")
    climatology = importlib.import_module(
        "ocean_solver.data.climatology" if args.layout == "legacy"
        else "ocean_solver.io.climatology"
    )
    entry.DEFAULT_CONFIG = replace(entry.DEFAULT_CONFIG, bathymetry_file=plan["data"]["bathymetry"]["path"])
    if "temperature" in plan["data"]:
        climatology.WOA_FILES = {role: plan["data"][role]["path"] for role in ("temperature", "salinity")}
    wind.CACHE_DIR = str(Path(plan["data"]["wind_900"]["path"]).parent)
    air.CACHE_DIR = str(Path(plan["data"]["air"]["path"]).parent)
    legacy_cache_adapter = args.layout == "legacy"
    if legacy_cache_adapter:
        # Historical readers capture CACHE_DIR as a function default. Pass the
        # selected cache explicitly, without editing frozen historical source.
        wind.load_monthly_wind = partial(wind.load_monthly_wind, cache_dir=wind.CACHE_DIR)
        entry.load_annual_mean_air_temp = partial(air.load_annual_mean_air_temp, cache_dir=air.CACHE_DIR)
        entry.load_monthly_mean_air_temp = partial(air.load_monthly_mean_air_temp, cache_dir=air.CACHE_DIR)
    argv = [*plan["argv"], "--tag", "probe", "--out-dir", str(directory / "model"),
            "--log-dir", str(directory / "logs")]
    sys.argv = ["global-regression-production", *argv]
    config = parse_run_configuration()
    selected = entry._input_files(config.args)
    if set(selected) != set(plan["data"]):
        raise ValueError("loader roles differ from frozen plan")
    for role, path in selected.items():
        if Path(path).resolve() != Path(plan["data"][role]["path"]).resolve():
            raise ValueError("loader selected a different file: " + role)
    source_scope = args.source_root.resolve() if args.source_root else package_root
    source_files = {str(path.relative_to(source_scope)): digest(path)
                    for path in sorted(source_scope.rglob("*.py"))}
    receipt = {
        "schema": "zhenmode.global_regression.v1", "execution_status": "running",
        "revision": args.revision, "layout": args.layout,
        "plan_sha256": digest(args.plan), "observer_sha256": digest(__file__),
        "package_root": str(package_root), "source_files": source_files,
        "legacy_cache_adapter": legacy_cache_adapter,
        "parsed_options": vars(config.args), "selected_data": plan["data"],
        "requested_steps": config.requested_steps,
        "timing_note": "first step includes lazy compilation; subsequent steps synchronized; observer hashing/IO excluded from step times; no isolated compiler timing",
    }
    write_json(directory / "started.json", receipt)
    class PreparedInputs(Exception):
        pass

    if args.prepare_only:
        native_initial_fields = entry.get_initial_fields

        def prepare_fields(grid):
            temperature, salinity = native_initial_fields(grid)
            with (directory / "initial-fields.npz").open("xb") as stream:
                np.savez(stream, T_init=temperature, S_init=salinity,
                         lon=grid.lon, lat=grid.lat, z=grid.z, wet_mask=grid.wet_mask)
            raise PreparedInputs()

        entry.get_initial_fields = prepare_fields
    native_integration = application.run_integration
    observed = []
    final = {}

    def state_identity(state):
        return {name: {"shape": list(value.shape), "dtype": value.dtype.str,
                       "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                       "finite": bool(np.isfinite(value).all())}
                for name, field in zip(state._fields, state, strict=True)
                for value in [np.asarray(field)]}

    def observe_integration(run_args, requested_steps, context, paths, recovery):
        receipt["setup_seconds"] = time.perf_counter() - started
        initial_record = {"step": recovery.start_step, "state": state_identity(recovery.state)}
        observed.append(initial_record)
        write_json(directory / f"step-{recovery.start_step:06d}.json", initial_record)
        native_step = context.solver.do_step

        def observe_step(state, day):
            step_started = time.perf_counter()
            updated = native_step(state, day)
            jax.block_until_ready(updated)
            elapsed = time.perf_counter() - step_started
            final["state"] = updated
            record = {"step": recovery.start_step + len(observed), "model_day_incoming": day,
                      "step_seconds": elapsed, "state": state_identity(updated)}
            observed.append(record)
            write_json(directory / f"step-{record['step']:06d}.json", record)
            return updated

        context.solver.do_step = observe_step
        return native_integration(run_args, requested_steps, context, paths, recovery)

    application.run_integration = observe_integration
    sys.argv = ["global-regression-production", *argv]
    loaded_caches = set()
    native_load = np.load

    def observe_load(path, *arguments, **kwargs):
        if isinstance(path, (str, Path)):
            loaded_caches.add(str(Path(path).resolve()))
        return native_load(path, *arguments, **kwargs)

    np.load = observe_load
    previous_stdout = sys.stdout
    try:
        code = entry.main()
    except PreparedInputs:
        code = 0
    finally:
        np.load = native_load
        if sys.stdout is not previous_stdout:
            sys.stdout.file.close()
            sys.stdout = previous_stdout
    expected_caches = {record["path"] for record in plan["data"].values()
                       if record["path"].endswith(".npz")}
    if not args.prepare_only and not expected_caches.issubset(loaded_caches):
        raise ValueError("frozen forcing caches were not actually read")
    receipt["actually_loaded_npz_paths"] = sorted(loaded_caches)
    if "state" in final:
        with (directory / "final-state.npz").open("xb") as stream:
            np.savez_compressed(stream, **{name: np.asarray(field)
                                         for name, field in zip(final["state"]._fields, final["state"], strict=True)})
    for record in plan["data"].values():
        if digest(record["path"]) != record["sha256"]:
            raise ValueError("input changed during execution: " + record["path"])
    if "restart" in plan and digest(plan["restart"]["path"]) != plan["restart"]["sha256"]:
        raise ValueError("restart input changed during execution")
    if source_files != {str(path.relative_to(source_scope)): digest(path)
                        for path in sorted(source_scope.rglob("*.py"))}:
        raise ValueError("execution source changed during run")
    loaded = {}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and Path(path).resolve().is_relative_to(source_scope):
            loaded[name] = {"path": str(Path(path).resolve()), "sha256": digest(path)}
    if args.prepare_only:
        complete = False
        receipt.update(stage="initial_preparation", accepted_steps=0, verdict="NOT_INTEGRATED",
                       duration_complete=False, initial_artifact_sha256=digest(directory / "initial-fields.npz"),
                       setup_seconds=time.perf_counter() - started)
    else:
        with np.load(directory / "model" / "global_probe.npz", allow_pickle=False) as result:
            complete = bool(result["duration_complete"])
            receipt.update(stage="integration", accepted_steps=int(result["accepted_steps"]), verdict=str(result["verdict"]),
                           duration_complete=complete, final_state_identity=json.loads(str(result["final_state_identity_json"])))
    receipt.update(execution_status="completed" if code == 0 and (complete or args.prepare_only) else "failed",
                   exit_code=code, observed_states=observed, loaded_source_files=loaded,
                   end_to_end_observer_seconds=time.perf_counter() - started)
    receipt["artifacts"] = {str(path.relative_to(directory)): digest(path)
                            for path in sorted(directory.rglob("*")) if path.is_file()}
    write_json(directory / "completed.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("execution_status", "accepted_steps", "verdict", "setup_seconds", "end_to_end_observer_seconds")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
