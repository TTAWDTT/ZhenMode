"""Independently constructed native rotating-wave arrays; no model/evaluator calls."""

import numpy as np

from tests.support.standing_wave import cgrid, fixture
from zhenmode.benchmarks.standing_wave import MOM6, OCEANANIGANS, digest


def planted(c, model):
    nx, ny, nz = c["nx"], c["ny"], c["nz"]
    time = np.arange(0, c["duration_s"] + 1, c["output_s"], dtype=float)
    dx, dy = c["Lx_m"] / nx, c["Ly_m"] / ny
    x, y = np.meshgrid((np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dy, indexing="ij")
    _, old = fixture()
    m = old["metadata"]
    m.update(
        schema=c["schema"],
        contract_sha256=digest(c),
        dt_s=c["dt"],
        steps=int(c["duration_s"] / c["dt"]),
        f=c["f"],
        Lx=c["Lx_m"],
        Ly=c["Ly_m"],
        model=model,
        zero_processes=c["explicitly_zero"],
    )
    if model != "ocean-solver":
        m.update(
            sampling_eta="cell_mean",
            sampling_u="node",
            velocity_layout="cgrid",
            source_sha=MOM6 if model == "MOM6" else OCEANANIGANS,
        )
    m["recorded_numerics"]["resolved_options"] = c[
        {
            "ocean-solver": "ocean_options",
            "MOM6": "mom_time_options",
            "Oceananigans": "oceananigans_options",
        }[model]
    ]
    shape = (len(time), nx * ny, nz)
    k = np.pi / c["Ly_m"]
    omega = 2 * np.pi / 32000
    factor = 1 if model == "ocean-solver" else np.sinc(1 / (2 * ny))
    eta = (
        0.01
        * np.cos(k * y.ravel())[None, :]
        * (0.25 + 0.75 * np.cos(omega * time))[:, None]
        * factor
    )
    h = np.broadcast_to(c["initial_h_m"], shape).copy()
    if model == "Oceananigans":
        h *= 1 + eta[..., None] / 100
    else:
        h[:, :, 0] += eta
    a = dict(
        metadata=m,
        time=time,
        x_eta=x.ravel(),
        y_eta=y.ravel(),
        area=np.full(nx * ny, dx * dy),
        eta=eta,
        h=h,
        T=np.full(shape, 15.0),
        S=np.full(shape, 35.0),
        x_u=x.ravel(),
        y_u=y.ravel(),
        x_v=x.ravel(),
        y_v=y.ravel(),
        width_u=np.zeros(nx * ny),
        u=np.zeros(shape),
        v=np.zeros(shape),
        volume_u=h * dx * dy,
        volume_v=h * dx * dy,
    )
    if model != "ocean-solver":
        a = cgrid(a, c)
    velocity_scale = 0.01 * np.sqrt(9.81 / 100) * np.sqrt(0.75)
    for q in ("u", "v"):
        signal = 0.5 * (1 - np.cos(omega * time)) if q == "u" else np.sin(omega * time)
        a[q] = np.broadcast_to(
            (velocity_scale * np.sin(k * a["y_" + q])[None, :] * signal[:, None])[..., None],
            a[q].shape,
        ).copy()
    return a
