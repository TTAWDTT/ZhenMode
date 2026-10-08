"""Pinned Julia environment, native wave execution and native-grid conversion."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import netCDF4
import numpy as np

from zhenmode.benchmarks.standing_wave import OCEANANIGANS, digest
from zhenmode.execution.resources import run_process_group
from zhenmode.execution.runs import write_json
from zhenmode.execution.standing_wave import metadata, native_arrays
from zhenmode.provenance.sources import load_json, sha256_file

PACKAGE = Path(__file__).parent


def pins():
    return load_json(PACKAGE / "pins.json")


def _files(root):
    """Actual upstream Julia implementation bytes, including extensions and project metadata."""
    root = Path(root)
    return {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in (".jl", ".toml")
    }


def _environment():
    return dict(
        os.environ,
        JULIA_NUM_THREADS="1",
        JULIA_NUM_PRECOMPILE_TASKS="1",
        JULIA_PKG_PRECOMPILE_AUTO="0",
        JULIA_LOAD_PATH="@:@stdlib",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )


def _invoke(julia, project, script, args, cwd, log, wall_seconds):
    command = [
        julia,
        "--startup-file=no",
        "--project=" + str(project),
        str(script),
        *map(str, args),
    ]
    with Path(log).open("x", encoding="utf8") as stream:
        done, resources = run_process_group(
            command,
            cwd=cwd,
            env=_environment(),
            stdout=stream,
            resources={"cpu": 1, "memory_mib": 4096, "wall_seconds": wall_seconds},
        )
    write_json(
        Path(str(log) + ".json"),
        dict(command=command, returncode=done.returncode, resources=resources),
        create=True,
    )
    if done.returncode:
        raise RuntimeError(f"native Julia command failed ({done.returncode}); see {log}")
    return resources


def doctor(cache, julia="julia", *, wall_seconds=600):
    """Capture the resolved runtime and require the selected upstream version and CUDA."""
    cache = Path(cache).resolve()
    if not (cache / "project/Manifest.toml").is_file():
        raise ValueError(
            "Oceananigans project/Manifest.toml is missing; prepare the pinned environment"
        )
    # Each inspection is append-only; receipt.json selects the current verified environment.
    attempts = cache / "inspections"
    attempts.mkdir(exist_ok=True)
    from uuid import uuid4

    attempt = attempts / uuid4().hex
    attempt.mkdir()
    project_hashes = {
        name: sha256_file(cache / "project" / name)
        for name in ("Project.toml", "Manifest.toml", "LocalPreferences.toml")
        if (cache / "project" / name).is_file()
    }
    _invoke(
        julia,
        cache / "project",
        PACKAGE / "runtime.jl",
        [attempt / "runtime.json"],
        attempt,
        attempt / "runtime.log",
        wall_seconds,
    )
    runtime = load_json(attempt / "runtime.json")
    selected = pins()
    if (
        runtime["oceananigans_version"] != selected["version"]
        or runtime["cuda_version"] != selected["cuda_version"]
    ):
        raise ValueError("resolved native package versions differ from pins; no silent downgrade")
    upstream_info = next(p for p in runtime["packages"].values() if p["name"] == "Oceananigans")
    if (
        upstream_info["tree_hash"] != selected["git_tree_sha1"]
        or runtime["actual_upstream_tree_hash"] != selected["git_tree_sha1"]
    ):
        raise ValueError("registered Oceananigans tree differs from pinned author commit")
    upstream = upstream_info["source"]
    sources = _files(upstream)
    if not sources or any(
        sha256_file(cache / "project" / name) != value for name, value in project_hashes.items()
    ):
        raise ValueError("native environment changed or upstream source is missing")
    program = Path(runtime["julia_executable"])
    info = dict(
        pins=selected,
        runtime=runtime,
        project_sha256=project_hashes,
        upstream_source_path=upstream,
        upstream_source_sha256=sources,
        julia_executable_sha256=sha256_file(program),
        driver_sha256=sha256_file(PACKAGE / "standing_wave.jl"),
        preparation_log=str(attempt / "runtime.log"),
        source_commit_provider="registered release pinned by version and recorded tree hash; actual source bytes verified separately",
    )
    write_json(cache / "receipt.json", info)
    return info


def prepare(cache, julia="julia", *, wall_seconds=900):
    """Resolve all explicit dependencies together and pin before any scientific run."""
    cache = Path(cache).resolve()
    if (cache / "receipt.json").exists():
        return verify(cache)
    project = cache / "project"
    project.mkdir(parents=True, exist_ok=True)
    # Keep provider selection local and inspect it alongside the manifest.
    preferences = project / "LocalPreferences.toml"
    if not preferences.exists():
        preferences.write_text('[FFTW]\nprovider = "fftw"\n', encoding="utf8")
    selected = pins()
    constraints = cache / "versions.toml"
    constraints.write_text(
        f'oceananigans = "{selected["version"]}"\ncuda = "{selected["cuda_version"]}"\n',
        encoding="utf8",
    )
    from uuid import uuid4

    log = cache / ("prepare-" + uuid4().hex + ".log")
    _invoke(julia, project, PACKAGE / "prepare.jl", [constraints], cache, log, wall_seconds)
    return doctor(cache, julia, wall_seconds=wall_seconds)


def verify(cache):
    cache = Path(cache).resolve()
    info = load_json(cache / "receipt.json")
    upstream_info = next(
        p for p in info["runtime"]["packages"].values() if p["name"] == "Oceananigans"
    )
    if (
        upstream_info["tree_hash"] != pins()["git_tree_sha1"]
        or info["runtime"]["actual_upstream_tree_hash"] != pins()["git_tree_sha1"]
    ):
        raise ValueError("native tree does not match the pinned upstream commit")
    if (
        info["pins"] != pins()
        or _files(info["upstream_source_path"]) != info["upstream_source_sha256"]
    ):
        raise ValueError("native Oceananigans source identity differs from receipt")
    if any(
        sha256_file(cache / "project" / name) != value
        for name, value in info["project_sha256"].items()
    ):
        raise ValueError("native project/manifest identity changed")
    if sha256_file(info["runtime"]["julia_executable"]) != info["julia_executable_sha256"]:
        raise ValueError("native Julia executable changed")
    return info


def integrate(config_file):
    """Called inside the suite's existing resource guard; launch no extra supervisor."""
    config = load_json(config_file)
    root = Path.cwd()
    cache = Path(config["oceananigans_cache"])
    info = verify(cache)
    driver = PACKAGE / "standing_wave.jl"
    identities = {
        "driver_sha256": sha256_file(driver),
        "configuration_sha256": sha256_file(config_file),
        "environment_receipt_sha256": sha256_file(cache / "receipt.json"),
    }
    command = [
        config["julia"],
        "--startup-file=no",
        "--project=" + str(cache / "project"),
        str(driver),
        str(config_file),
    ]
    write_json(
        root / "native-run.json",
        dict(environment=info, inputs=identities, command=command),
        create=True,
    )
    subprocess.run(command, cwd=root, env=_environment(), check=True)
    verify(cache)
    if (
        sha256_file(driver) != identities["driver_sha256"]
        or sha256_file(config_file) != identities["configuration_sha256"]
    ):
        raise ValueError("native driver/configuration changed during execution")


def convert(c, directory, resources):
    root = Path(directory)
    run = load_json(root / "native-run.json")
    env = run["environment"]
    with netCDF4.Dataset(root / "native.nc") as native:
        time = np.asarray(native["time"][:])
        if np.ma.is_masked(native["time"][:]) or not np.array_equal(
            time, np.arange(0, c["period_s"] + 1, c["output_s"])
        ):
            raise ValueError("native Oceananigans did not save all required actual times")
        fields = {"time": time}
        for name in ("eta", "h", "T", "S", "u", "v"):
            value = native[name][:]
            if np.ma.is_masked(value):
                raise ValueError("missing native Oceananigans state: " + name)
            if name == "eta":
                fields[name] = np.asarray(value).transpose(2, 0, 1)
            else:
                fields[name] = np.asarray(value).transpose(3, 0, 1, 2)[..., ::-1].copy()
        x, y = np.meshgrid(native["x_eta"][:], native["y_eta"][:], indexing="ij")
        ux, uy = np.meshgrid(native["x_u"][:], native["y_u"][:], indexing="ij")
        vx, vy = np.meshgrid(native["x_v"][:], native["y_v"][:], indexing="ij")
        initialization_s = float(native.initialization_s)
        integration_s = float(native.integration_s)
        if (
            int(native.accepted_steps) != int(c["period_s"] / c["dt"])
            or native.vertical_order != "bottom_to_top"
        ):
            raise ValueError("native step count or layer order differs from frozen case")
        native_details = {
            "model": native.model_description,
            "substepping": native.substepping,
            "thermodynamic_variables": "native linear-EOS anomalies plus two advected physical witnesses",
        }
    write_json(
        root / "resolved-options.json",
        dict(requested=c["oceananigans_options"], actual=native_details),
        create=True,
    )
    information = metadata(
        c,
        "Oceananigans",
        source_sha=OCEANANIGANS,
        executable_sha256=env["julia_executable_sha256"],
        input_sha256=run["inputs"]["configuration_sha256"],
        config_sha256=digest(
            {
                "contract": c,
                "environment": env["project_sha256"],
                "driver": run["inputs"]["driver_sha256"],
            }
        ),
        initialization_s=initialization_s,
        integration_s=integration_s,
        resources=resources,
        numerics=dict(
            time_scheme="native SplitRungeKutta3 with split-explicit ForwardBackward free surface",
            transport="full momentum; T/S anomalies and two nonzero physical scalar witnesses advected",
            filters="native split-explicit averaging kernel, retained without tuning",
            vertical_coordinate="native z-star scaling; actual grid thickness exported",
            substeps=native_details["substepping"],
            resolved_options=c["oceananigans_options"],
        ),
    )
    return native_arrays(
        c, fields, layout="cgrid", x_eta=x, y_eta=y, x_u=ux, y_u=uy, x_v=vx, y_v=vy
    ), information


def main(argv=None):
    parser = argparse.ArgumentParser(prog="zhenmode baseline oceananigans")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "doctor"):
        p = commands.add_parser(command)
        p.add_argument("--cache", required=True)
        p.add_argument("--julia", default="julia")
        p.add_argument("--wall-seconds", type=int, default=900 if command == "prepare" else 600)
    args = parser.parse_args(argv)
    result = (prepare if args.command == "prepare" else doctor)(
        args.cache, args.julia, wall_seconds=args.wall_seconds
    )
    import json

    print(
        json.dumps(
            {
                "pins": result["pins"],
                "runtime": result["runtime"]["device"],
                "project_sha256": result["project_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-config", required=True)
    integrate(parser.parse_args().worker_config)
