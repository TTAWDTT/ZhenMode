"""Wave adapter for the existing complete FD core; no alternative integrator."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
import time
from pathlib import Path

import numpy as np

from zhenmode.benchmarks.standing_wave import digest
from zhenmode.execution.runs import environment_identity, write_json
from zhenmode.execution.standing_wave import metadata, native_arrays
from zhenmode.provenance.sources import load_json, package_source_hashes, sha256_file

ZERO_COEFFICIENTS = (
    "nu_h",
    "nu_v",
    "nu_bi",
    "kappa_h",
    "kappa_v",
    "kappa_bi",
    "kappa_conv",
    "kappa_gm",
    "kappa_redi",
    "r_bot",
    "cd",
)


def runtime_identity():
    """Identify the real interpreter, dependency versions and loaded native JAX implementation."""
    import jaxlib

    info = environment_identity()
    interpreter = Path(sys._base_executable).resolve()
    info["actual_executable"] = str(interpreter)
    info["actual_executable_sha256"] = sha256_file(interpreter)
    info["native_jaxlib_sha256"] = {
        path.relative_to(Path(jaxlib.__file__).parent).as_posix(): sha256_file(path)
        for path in sorted(Path(jaxlib.__file__).parent.rglob("*"))
        if path.is_file() and path.suffix in (".so", ".pyd", ".dll")
    }
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name") or ""
        if name.lower().startswith(("jax-cuda", "nvidia-")):
            info["packages"][name] = dist.version
    return info


def integrate(config_file):
    from zhenmode.model.config import PhysicsConfig
    from zhenmode.model.solver.factory import make_solver_global
    from zhenmode.model.solver.geometry.grid import GlobalOceanGrid
    from zhenmode.model.solver.numerics.backend import jax, jnp

    if jax.default_backend() != "gpu":
        raise RuntimeError("ZhenMode wave execution requires actual CUDA")
    config = load_json(config_file)
    c, root = config["contract"], Path.cwd()
    sources = package_source_hashes(__file__)
    runtime = runtime_identity()
    nx, ny, nz = c["nx"], c["ny"], c["nz"]
    dx, dy = c["Lx_m"] / nx, c["Ly_m"] / ny
    x, y = (np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dy
    z = -np.linspace(0, c["H_m"], nz)
    grid = GlobalOceanGrid(
        lon=x,
        lat=y,
        dx_2d=np.full((nx, ny), dx),
        dy=dy,
        cos_lat=np.ones(ny),
        f=np.zeros((nx, ny)),
        z=z,
        dz=-np.diff(z),
        nz=nz,
        depth=np.full((nx, ny), c["H_m"]),
        wet_mask=np.ones((nx, ny)),
        ocean_mask=np.ones((nx, ny), bool),
        land_mask=np.zeros((nx, ny)),
        wet_mask_3d=np.ones((nx, ny, nz)),
        nx=nx,
        ny=ny,
    )
    physics = PhysicsConfig(**dict.fromkeys(ZERO_COEFFICIENTS, 0.0))
    begin = time.monotonic()
    step, initialize, _, params, _ = make_solver_global(
        grid, physics, dt=c["dt"], return_params=True, **c["ocean_options"]
    )
    state = initialize()._replace(
        eta=jnp.asarray(
            np.broadcast_to(c["amplitude_m"] * np.cos(2 * np.pi * x / c["Lx_m"])[:, None], (nx, ny))
        )
    )
    jax.block_until_ready(state)
    with (root / "initial.npz").open("xb") as stream:
        np.savez_compressed(stream, **state._asdict())
    tick = time.monotonic()
    compiled = step.lower(state).compile()
    compilation_s = time.monotonic() - tick
    initialization_s = time.monotonic() - begin
    history = []

    def save():
        history.append(
            {name: np.asarray(getattr(state, name)).copy() for name in ("u", "v", "T", "S", "eta")}
        )

    save()
    tick = time.monotonic()
    steps = int(c["period_s"] / c["dt"])
    every = int(c["output_s"] / c["dt"])
    for index in range(1, steps + 1):
        state = compiled(state)
        jax.block_until_ready(state)
        if index % every == 0:
            if not all(np.isfinite(np.asarray(v)).all() for v in state):
                raise ValueError("nonfinite native FD state")
            save()
            print(f"accepted_step={index} simulation_seconds={index * c['dt']}", flush=True)
    integration_s = time.monotonic() - tick
    arrays = {name: np.stack([row[name] for row in history]) for name in history[0]}
    with (root / "native.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            **arrays,
            time=np.arange(0, c["period_s"] + 1, c["output_s"], dtype=float),
            x=x,
            y=y,
            dz_node=np.asarray(params.dz_node),
        )
    if sources != package_source_hashes(__file__):
        raise ValueError("FD runtime sources changed")
    write_json(
        root / "native-run.json",
        dict(
            source_sha=config["source_revision"],
            source_files_sha256=sources,
            runtime=runtime,
            backend=jax.default_backend(),
            device=str(jax.devices()[0]),
            initialization_s=initialization_s,
            compilation_s=compilation_s,
            integration_s=integration_s,
            input_sha256=sha256_file(root / "initial.npz"),
            physics_zero={name: getattr(physics, name) for name in ZERO_COEFFICIENTS},
            actual_controls=c["ocean_options"],
            coordinate_system="Cartesian metres in coordinate slots; exact physical metrics",
        ),
        create=True,
    )


def convert(c, directory, resources):
    root = Path(directory)
    run = load_json(root / "native-run.json")
    with np.load(root / "native.npz", allow_pickle=False) as native:
        fields = {name: native[name].copy() for name in ("time", "u", "v", "T", "S", "eta")}
        x, y = np.meshgrid(native["x"], native["y"], indexing="ij")
        h = np.broadcast_to(native["dz_node"], fields["T"].shape).copy()
    h[..., 0] += fields["eta"]
    fields["h"] = h
    information = metadata(
        c,
        "ocean-solver",
        source_sha=run["source_sha"],
        executable_sha256=run["runtime"]["actual_executable_sha256"],
        input_sha256=run["input_sha256"],
        config_sha256=digest(
            {
                "contract": c,
                "controls": run["actual_controls"],
                "physics": run["physics_zero"],
                "runtime": run["runtime"],
            }
        ),
        initialization_s=run["initialization_s"],
        integration_s=run["integration_s"],
        resources=resources,
        numerics=dict(
            time_scheme="existing L/N/L plus forward-backward free surface, legacy full step, CUDA float64",
            transport="full momentum and T/S transport active",
            filters="native filtering; no cap or sponge",
            vertical_coordinate="fixed nodal dual cells; h includes diagnostic surface displacement",
            substeps="native frozen per-resolution steps",
            resolved_options=run["actual_controls"],
        ),
    )
    return native_arrays(c, fields, layout="collocated", x_eta=x, y_eta=y), information


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-config", required=True)
    integrate(parser.parse_args().worker_config)
