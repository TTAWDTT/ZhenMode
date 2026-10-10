"""Shared native benchmark serialization and bounded process supervision."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from zhenmode.benchmarks.standing_wave import digest
from zhenmode.evaluation.native_channel import duration
from zhenmode.execution.resources import run_cuda_worker, run_process_group
from zhenmode.execution.runs import write_json
from zhenmode.provenance.sources import package_source_hashes, sha256_file, source_root


def metadata(
    c,
    model,
    *,
    source_sha,
    executable_sha256,
    input_sha256,
    config_sha256,
    initialization_s,
    integration_s,
    resources,
    numerics,
):
    sampling = c["native_sampling"][model]
    return dict(
        schema=c["schema"],
        contract_sha256=digest(c),
        model=model,
        source_sha=source_sha,
        executable_sha256=executable_sha256,
        input_sha256=input_sha256,
        config_sha256=config_sha256,
        coordinate_system="cartesian_m",
        units=dict(
            time="s", x="m", eta="m", h="m", u="m/s", volume="m3", area="m2", T="degC", S="psu"
        ),
        snapshot_kind="instantaneous",
        sampling_eta=sampling["eta"],
        sampling_u=sampling["u"],
        velocity_layout=sampling["layout"],
        dt_s=c["dt"],
        steps=int(duration(c) / c["dt"]),
        full_dynamics=True,
        zero_processes=c["explicitly_zero"],
        recorded_numerics=numerics,
        initialization_s=initialization_s,
        integration_s=integration_s,
        total_wall_s=resources["elapsed_wall_seconds"],
        aggregate_peak_bytes=resources["peak_host_rss_bytes"],
        cpu=1,
        ranks=1,
        trajectory=dict(kind="continuous", processes=1),
        g=c["gravity"],
        rho0=c["rho0"],
        H=c["H_m"],
        Lx=c["Lx_m"],
        Ly=c["Ly_m"],
        f=c["f"],
        eos=c["eos"],
        boundary=c["boundary"],
    )


def native_arrays(c, fields, *, layout, x_eta, y_eta, x_u=None, y_u=None, x_v=None, y_v=None):
    """Serialize native states without regridding; evaluator checks geometry independently."""
    nt, nx, ny, nz = len(fields["time"]), c["nx"], c["ny"], c["nz"]
    dx, dy = c["Lx_m"] / nx, c["Ly_m"] / ny
    h = fields["h"]
    if layout == "collocated":
        x_u, y_u, x_v, y_v = x_eta, y_eta, x_eta, y_eta
        vu = vv = h * dx * dy
    else:
        vu = 0.5 * (h + np.roll(h, 1, axis=1)) * dx * dy
        vv = (
            np.concatenate(
                (0.5 * h[:, :, :1], 0.5 * (h[:, :, :-1] + h[:, :, 1:]), 0.5 * h[:, :, -1:]), axis=2
            )
            * dx
            * dy
        )
    return dict(
        time=fields["time"],
        x_eta=x_eta.ravel(),
        y_eta=y_eta.ravel(),
        area=np.full(nx * ny, dx * dy),
        eta=fields["eta"].reshape(nt, -1),
        **{name: fields[name].reshape(nt, nx * ny, nz) for name in ("h", "T", "S")},
        x_u=x_u.ravel(),
        y_u=y_u.ravel(),
        width_u=np.zeros(x_u.size),
        u=fields["u"].reshape(nt, -1, nz),
        volume_u=vu.reshape(nt, -1, nz),
        x_v=x_v.ravel(),
        y_v=y_v.ravel(),
        v=fields["v"].reshape(nt, -1, nz),
        volume_v=vv.reshape(nt, -1, nz),
    )


def publish(c, arrays, information, directory, evaluator):
    arrays["metadata"] = information
    report = evaluator(c, arrays)
    path = Path(directory) / "output.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **(arrays | {"metadata": np.array(json.dumps(information))}))
    report["output_sha256"] = sha256_file(path)
    write_json(Path(directory) / "score.json", report, create=True)
    return report


def launch(config, model, directory, *, wall_seconds=600):
    """One owned CPU/process group; ZhenMode and Oceananigans require CUDA."""
    command = [
        sys.executable,
        "-m",
        {
            "zhenmode": "zhenmode.execution.wave_zhenmode",
            "mom6": "zhenmode.baselines.mom6.standing_wave",
            "oceananigans": "zhenmode.baselines.oceananigans.adapter",
        }[model],
        "--worker-config",
        str(config),
    ]
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        JULIA_NUM_THREADS="1",
        JULIA_NUM_PRECOMPILE_TASKS="1",
    )
    if model != "mom6":
        env.update(
            JAX_PLATFORMS="cuda",
            CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            XLA_PYTHON_CLIENT_MEM_FRACTION=".40",
            XLA_FLAGS="--xla_gpu_autotune_level=0",
        )
    supervise = run_cuda_worker if model == "zhenmode" else run_process_group
    with (directory / "run.log").open("x", encoding="utf8") as log:
        result, resources = supervise(
            command,
            cwd=directory,
            env=env,
            stdout=log,
            resources={"cpu": 1, "memory_mib": 4096, "wall_seconds": wall_seconds},
        )
    write_json(
        directory / "resources.json",
        dict(returncode=result.returncode, resources=resources),
        create=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"{model} worker failed ({result.returncode}); see {directory / 'run.log'}"
        )
    return resources


MODELS = ("zhenmode", "mom6", "oceananigans")


def execute(
    c,
    output,
    *,
    evaluator,
    definition_validator,
    reporter,
    scope,
    limitations,
    models=("zhenmode", "mom6", "oceananigans"),
    mom_executable=None,
    mom_source=None,
    oceananigans_cache=None,
    julia="julia",
    source_revision=None,
):
    if os.name != "posix":
        raise ValueError("native channel execution requires Linux/WSL and CUDA")
    if not models or len(models) != len(set(models)) or any(m not in MODELS for m in models):
        raise ValueError("select distinct supported models")
    if "mom6" in models and (mom_executable is None or mom_source is None):
        raise ValueError("MOM6 executable and pinned source directory are required")
    if "oceananigans" in models and oceananigans_cache is None:
        raise ValueError("a prepared Oceananigans environment is required")
    revision = reported_revision(source_revision)
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("source-revision must be a full Git revision")
    case = c["case"]
    definition_validator(c)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = dict(
        contract=c,
        source_revision=revision,
        source_revision_provider="reported revision; runtime package bytes recorded separately",
        mom_executable=str(Path(mom_executable).resolve()) if mom_executable else None,
        mom_source=str(Path(mom_source).resolve()) if mom_source else None,
        oceananigans_cache=str(Path(oceananigans_cache).resolve()) if oceananigans_cache else None,
        julia=julia,
    )
    write_json(root / "contract.json", c, create=True)
    write_json(root / "configuration.json", config, create=True)
    sources = package_source_hashes(__file__)
    receipt = dict(
        case=case,
        benchmark=c.get("benchmark", "standing-wave"),
        status="running",
        completed_models=[],
        models={},
        package_source_sha256=sources,
        performance_comparison=False,
        industrial_qualified=False,
    )
    try:
        for model in models:
            directory = root / model
            directory.mkdir()
            receipt["current_model"] = model
            write_json(root / "run.json", receipt)
            resources = launch(root / "configuration.json", model, directory)
            from zhenmode.baselines.mom6.standing_wave import convert as convert_mom
            from zhenmode.baselines.oceananigans.adapter import convert as convert_oceananigans
            from zhenmode.evaluation.native_channel import load
            from zhenmode.execution.wave_zhenmode import convert as convert_zhenmode

            convert = {
                "mom6": convert_mom,
                "zhenmode": convert_zhenmode,
                "oceananigans": convert_oceananigans,
            }[model]
            arrays, information = convert(c, directory, resources)
            report = publish(c, arrays, information, directory, evaluator)
            # Verify the published representation also survives the strict reader.
            evaluator(c, load(directory / "output.npz"))
            receipt["models"][model] = report
            receipt["completed_models"].append(model)
        if package_source_hashes(__file__) != sources:
            raise ValueError("executed package changed during comparison")
        receipt.update(
            status="completed",
            comparison_scope=scope,
            limitations=limitations,
        )
        write_json(root / "comparison.json", receipt, create=True)
        reporter(root / "comparison.md", receipt)
    except BaseException as error:
        receipt.update(status="failed", reason=type(error).__name__ + ": " + str(error))
        raise
    finally:
        write_json(root / "run.json", receipt)
    return receipt


def reported_revision(explicit=None):
    """Resolve only this package's checkout; unrelated cwd repositories cannot identify it."""
    if explicit is not None:
        return explicit
    checkout = source_root(__file__).parent
    if not (checkout / ".git").exists():
        raise ValueError(
            "installed runs require --source-revision; actual package bytes are also recorded"
        )
    try:
        return subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise ValueError(
            "cannot read package checkout revision; supply --source-revision"
        ) from error
