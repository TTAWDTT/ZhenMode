"""Independent closed-form wave arrays for scorer controls; never model outputs."""

import numpy as np

from zhenmode.benchmarks import standing_wave as case_definition


def fixture(case="coarse", half=False, sampling="node"):
    c = case_definition.legacy_contract(case, half)
    nx, ny, nz = c["nx"], c["ny"], c["nz"]
    dx, dy = c["Lx_m"] / nx, c["Ly_m"] / ny
    x, y = np.meshgrid((np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dy, indexing="ij")
    x, y = x.ravel(), y.ravel()
    t = np.arange(0, 32001, 1000, dtype=float)
    # Derived separately from eta_tt = g H eta_xx, with omega=2pi/32000.
    # Do not call the scorer's exact() implementation to generate its test truth.
    phase = 2 * np.pi * t / 32000.0
    spatial_phase = 2 * np.pi * x / (32000 * np.sqrt(9.81 * 100.0))
    factor = 1.0 if sampling == "node" else np.sinc(1 / nx)
    eta = c["amplitude_m"] * np.cos(spatial_phase)[None, :] * np.cos(phase)[:, None] * factor
    ubar = (
        c["amplitude_m"]
        * np.sqrt(9.81 / 100.0)
        * np.sin(spatial_phase)[None, :]
        * np.sin(phase)[:, None]
        * factor
    )
    shape = (len(t), nx * ny, nz)
    h = np.broadcast_to(c["initial_h_m"], shape).copy()
    h[:, :, 0] += eta
    vol = h * dx * dy
    m = dict(
        schema=case_definition.SCHEMA_V0,
        contract_sha256=case_definition.digest(c),
        model="ocean-solver",
        source_sha="a" * 40,
        executable_sha256="b" * 64,
        input_sha256="c" * 64,
        config_sha256="d" * 64,
        coordinate_system="cartesian_m",
        units=dict(
            time="s", x="m", eta="m", h="m", u="m/s", volume="m3", area="m2", T="degC", S="psu"
        ),
        snapshot_kind="instantaneous",
        sampling_eta=sampling,
        sampling_u=sampling,
        velocity_layout="collocated",
        dt_s=c["dt"],
        steps=int(32000 / c["dt"]),
        full_dynamics=True,
        zero_processes=c["explicitly_zero"].copy(),
        recorded_numerics={
            k: "synthetic oracle only"
            for k in ["time_scheme", "transport", "filters", "vertical_coordinate", "substeps"]
        },
        initialization_s=1.0,
        integration_s=2.0,
        total_wall_s=3.0,
        aggregate_peak_bytes=1024,
        cpu=1,
        ranks=1,
        trajectory=dict(kind="continuous", processes=1),
        g=9.81,
        rho0=1025.0,
        H=100.0,
        Lx=c["Lx_m"],
        Ly=c["Ly_m"],
        f=0.0,
        eos=c["eos"],
        boundary=c["boundary"],
    )
    m["recorded_numerics"]["resolved_options"] = c["ocean_options"].copy()
    a = dict(
        metadata=m,
        time=t,
        x_eta=x,
        y_eta=y,
        area=np.full(nx * ny, dx * dy),
        eta=eta,
        h=h,
        T=np.full(shape, 15.0),
        S=np.full(shape, 35.0),
        x_u=x.copy(),
        y_u=y.copy(),
        width_u=np.full(x.size, 0.0 if sampling == "node" else dx),
        u=np.broadcast_to(ubar[..., None], shape).copy(),
        volume_u=vol.copy(),
        x_v=x.copy(),
        y_v=y.copy(),
        v=np.zeros(shape),
        volume_v=vol.copy(),
    )
    return c, a


def cgrid(a, c):
    nx, ny, nz = c["nx"], c["ny"], c["nz"]
    nt = len(a["time"])
    dx, dy = c["Lx_m"] / nx, c["Ly_m"] / ny
    h = a["h"].reshape(nt, nx, ny, nz)
    for q, xs, ys in [
        ("u", np.arange(nx) * dx, (np.arange(ny) + 0.5) * dy),
        ("v", (np.arange(nx) + 0.5) * dx, np.arange(ny + 1) * dy),
    ]:
        x, y = np.meshgrid(xs, ys, indexing="ij")
        a["x_" + q], a["y_" + q] = x.ravel(), y.ravel()
    a["width_u"] = np.zeros(nx * ny)
    phase = 2 * np.pi * a["time"] / 32000.0
    spatial_phase = 2 * np.pi * a["x_u"] / (32000 * np.sqrt(9.81 * 100.0))
    u = (
        c["amplitude_m"]
        * np.sqrt(9.81 / 100.0)
        * np.sin(spatial_phase)[None, :]
        * np.sin(phase)[:, None]
    )
    a["u"] = np.broadcast_to(u[..., None], (nt, nx * ny, nz)).copy()
    a["v"] = np.zeros((nt, nx * (ny + 1), nz))
    a["volume_u"] = (0.5 * (h + np.roll(h, 1, axis=1)) * dx * dy).reshape(nt, nx * ny, nz)
    a["volume_v"] = (
        np.concatenate(
            (0.5 * h[:, :, :1], 0.5 * (h[:, :, :-1] + h[:, :, 1:]), 0.5 * h[:, :, -1:]), axis=2
        )
        * dx
        * dy
    ).reshape(nt, nx * (ny + 1), nz)
    a["metadata"]["velocity_layout"] = "cgrid"
    a["metadata"]["sampling_u"] = "node"
    return a
