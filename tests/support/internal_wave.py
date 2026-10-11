"""Independent analytic internal-mode arrays for scorer and producer controls.

No production internal-wave initializer/reference is imported. The root is
bisected from the EOS and stratification; cell/layer means use antiderivatives,
not the producer's sinc implementation. Synthetic future layer geometry is a
legal scorer input, not a prediction of native MOM6 layer evolution.
"""

import copy
import math

import numpy as np

from tests.support.standing_wave import fixture as metadata_fixture
from zhenmode.benchmarks.standing_wave import digest

MODEL_NAMES = {"zhenmode": "ocean-solver", "mom6": "MOM6", "oceananigans": "Oceananigans"}
OPTIONS = {"ocean-solver": "ocean_options", "MOM6": "mom_time_options", "Oceananigans": "oceananigans_options"}


def parameters(c):
    """Return an independently computed first internal root and frequency."""
    alpha = -c["eos"]["drho_dT"] / c["rho0"]
    gradient = c["internal_wave"]["dT_dz_C_m"]
    stratification = c["gravity"] * alpha * gradient
    beta = stratification * c["H_m"] / c["gravity"]
    left, right = math.pi, 1.5 * math.pi - 1e-8
    assert left * math.tan(left) < beta < right * math.tan(right)
    for _ in range(100):
        middle = (left + right) / 2
        if middle == left or middle == right:
            break
        if middle * math.tan(middle) < beta:
            left = middle
        else:
            right = middle
    root = (left + right) / 2
    vertical = root / c["H_m"]
    horizontal = math.pi / c["Ly_m"]
    frequency = math.sqrt(c["f"] ** 2 + stratification * horizontal ** 2 / vertical ** 2)
    return root, vertical, horizontal, frequency


def _edges(surface, widths):
    return np.concatenate((surface[..., None], surface[..., None] - np.cumsum(widths, axis=-1)), axis=-1)


def _vertical_means(edges, m, H):
    upper, lower = edges[..., :-1], edges[..., 1:]
    thickness = upper - lower
    sine = (np.cos(m * (lower + H)) - np.cos(m * (upper + H))) / (m * thickness)
    cosine = (np.sin(m * (upper + H)) - np.sin(m * (lower + H))) / (m * thickness)
    return (upper + lower) / 2, sine, cosine


def planted(c, model=None):
    """Return complete synthetic native arrays for any of the 21 declared rows."""
    model = MODEL_NAMES[c["frozen_model"]] if model is None else MODEL_NAMES.get(model, model)
    assert model == MODEL_NAMES[c["frozen_model"]]
    _, template = metadata_fixture()
    metadata = copy.deepcopy(template["metadata"])
    nx, ny, nz = c["nx"], c["ny"], c["nz"]
    dx, dy = c["Lx_m"] / nx, c["Ly_m"] / ny
    times = np.arange(0, c["duration_s"] + 1, c["output_s"], dtype=float)
    nt = len(times)
    x, y = np.meshgrid((np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dy, indexing="ij")
    root, m, k, omega = parameters(c)
    A, gamma = c["amplitude_m"], c["internal_wave"]["dT_dz_C_m"]
    cosine_time = np.cos(omega * times)[:, None, None]
    sine_time = np.sin(omega * times)[:, None, None]
    if model == "ocean-solver":
        horizontal_scalar = np.cos(k * y)
    else:
        horizontal_scalar = (np.sin(k * (y + dy / 2)) - np.sin(k * (y - dy / 2))) / (k * dy)
    eta = A * math.sin(root) * horizontal_scalar[None, :, :] * cosine_time
    widths = np.broadcast_to(c["initial_h_m"], (nt, nx, ny, nz)).copy()
    if model == "Oceananigans":
        widths *= 1 + eta[..., None] / c["H_m"]
    else:
        widths[..., 0] += eta
    area = np.full(nx * ny, dx * dy)
    if model == "ocean-solver":
        depth = np.broadcast_to(-np.linspace(0, c["H_m"], nz), widths.shape)
        sine_z, cosine_z = np.sin(m * (depth + c["H_m"])), np.cos(m * (depth + c["H_m"]))
        T = c["T_C"] + gamma * depth - gamma * A * sine_z * horizontal_scalar[None, ..., None] * cosine_time[..., None]
        u = -A * c["f"] * m / k * cosine_z * np.sin(k * y)[None, ..., None] * cosine_time[..., None]
        v = A * omega * m / k * cosine_z * np.sin(k * y)[None, ..., None] * sine_time[..., None]
        xu, yu, xv, yv = x, y, x, y
        volume_u = volume_v = widths * dx * dy
    else:
        edges = _edges(eta, widths)
        depth, sine_z, _ = _vertical_means(edges, m, c["H_m"])
        T = c["T_C"] + gamma * depth - gamma * A * sine_z * horizontal_scalar[None, ..., None] * cosine_time[..., None]
        xu, yu = np.meshgrid(np.arange(nx) * dx, (np.arange(ny) + 0.5) * dy, indexing="ij")
        xv, yv = np.meshgrid((np.arange(nx) + 0.5) * dx, np.arange(ny + 1) * dy, indexing="ij")
        uedges = (edges + np.roll(edges, 1, axis=1)) / 2
        vedges = np.concatenate((edges[:, :, :1], (edges[:, :, :-1] + edges[:, :, 1:]) / 2, edges[:, :, -1:]), axis=2)
        _, _, ucos = _vertical_means(uedges, m, c["H_m"])
        _, _, vcos = _vertical_means(vedges, m, c["H_m"])
        u = -A * c["f"] * m / k * ucos * np.sin(k * yu)[None, ..., None] * cosine_time[..., None]
        v = A * omega * m / k * vcos * np.sin(k * yv)[None, ..., None] * sine_time[..., None]
        volume_u = (widths + np.roll(widths, 1, axis=1)) / 2 * dx * dy
        volume_v = np.concatenate((widths[:, :, :1] / 2, (widths[:, :, :-1] + widths[:, :, 1:]) / 2, widths[:, :, -1:] / 2), axis=2) * dx * dy
    metadata.update(
        schema=c["schema"], contract_sha256=digest(c), model=model,
        dt_s=c["dt"], steps=int(c["duration_s"] / c["dt"]),
        sampling_eta=c["native_sampling"][model]["eta"],
        sampling_u=c["native_sampling"][model]["u"],
        velocity_layout=c["native_sampling"][model]["layout"],
        zero_processes=c["explicitly_zero"].copy(),
        g=c["gravity"], rho0=c["rho0"], H=c["H_m"], Lx=c["Lx_m"], Ly=c["Ly_m"],
        f=c["f"], eos=copy.deepcopy(c["eos"]), boundary=copy.deepcopy(c["boundary"]),
    )
    if model == "MOM6":
        metadata["source_sha"] = "f49a00096df607b48354603e2398e14e189fd62e"
    elif model == "Oceananigans":
        metadata["source_sha"] = c["oceananigans_options"]["source_commit"]
    metadata["recorded_numerics"]["resolved_options"] = copy.deepcopy(c[OPTIONS[model]])
    return dict(
        metadata=metadata, time=times, x_eta=x.ravel(), y_eta=y.ravel(), area=area,
        eta=eta.reshape(nt, -1), h=widths.reshape(nt, nx * ny, nz),
        T=T.reshape(nt, nx * ny, nz), S=np.full((nt, nx * ny, nz), c["S_psu"]),
        x_u=xu.ravel(), y_u=yu.ravel(), width_u=np.zeros(xu.size), u=u.reshape(nt, -1, nz),
        volume_u=volume_u.reshape(nt, -1, nz),
        x_v=xv.ravel(), y_v=yv.ravel(), v=v.reshape(nt, -1, nz),
        volume_v=volume_v.reshape(nt, -1, nz),
    )


def fixture(c, model=None):
    """Alias returning arrays; the caller supplies its declared contract."""
    return planted(c, model)
