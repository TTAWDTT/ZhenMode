"""Native-position references for rotating adjustment and stratified thermal wind."""

import argparse
import json
from pathlib import Path

import numpy as np

from zhenmode.benchmarks.channel_dynamics import validate_contract
from zhenmode.benchmarks.standing_wave import digest
from zhenmode.evaluation.native_channel import load, require
from zhenmode.evaluation.native_channel import validate as validate_native
from zhenmode.execution.runs import write_json
from zhenmode.provenance.sources import sha256_file


def temperature_front(c, y):
    # Reference derivation is separate from the native initialization producer.
    o = c["thermal_wind"]
    angle = np.pi * (np.asarray(y) - c["Ly_m"] / 2) / (2 * o["front_width_m"])
    delta = o["delta_T_C"] * np.sin(np.clip(angle, -np.pi / 2, np.pi / 2))
    slope = np.where(
        np.abs(angle) < np.pi / 2,
        o["delta_T_C"] * np.pi / (2 * o["front_width_m"]) * np.cos(angle),
        0,
    )
    return delta, slope


def reference_surface(c, time, y, width=0):
    y = np.asarray(y)
    time = np.asarray(time)
    if c["benchmark"] == "geostrophic-adjustment":
        k = np.pi / c["Ly_m"]
        omega = np.sqrt(c["f"] ** 2 + c["gravity"] * c["H_m"] * k * k)
        fraction = c["f"] ** 2 / (omega * omega)
        return (
            c["amplitude_m"]
            * np.cos(k * y)[None, :]
            * np.sinc(k * width / (2 * np.pi))
            * (fraction + (1 - fraction) * np.cos(omega * time))[:, None]
        )
    nodes, weights = np.polynomial.legendre.leggauss(32)
    delta, _ = temperature_front(c, y[..., None] + width * nodes / 2)
    alpha = -c["eos"]["drho_dT"] / c["rho0"]
    gamma = c["thermal_wind"]["dT_dz_C_m"]
    eta = alpha * c["H_m"] * delta
    for _ in range(5):
        residual = eta - alpha * gamma * eta**2 / 2 - alpha * delta * (eta + c["H_m"])
        eta -= residual / (1 - alpha * gamma * eta - alpha * delta)
    eta = np.sum(eta * weights / 2, axis=-1)
    return np.broadcast_to(eta, (len(time), len(y))).copy()


def reference_velocity(c, time, y, z, component):
    time = np.asarray(time)
    y = np.asarray(y)
    if c["benchmark"] == "geostrophic-adjustment":
        k = np.pi / c["Ly_m"]
        omega = np.sqrt(c["f"] ** 2 + c["gravity"] * c["H_m"] * k * k)
        scale = c["amplitude_m"] * c["gravity"] * k / omega
        signal = (
            np.sin(omega * time)
            if component == "v"
            else c["f"] / omega * (1 - np.cos(omega * time))
        )
        return np.broadcast_to(
            (scale * signal[:, None] * np.sin(k * y)[None, :])[..., None], z.shape
        ).copy()
    if component == "v":
        return np.zeros_like(z)
    _, gradient = temperature_front(c, y)
    alpha = -c["eos"]["drho_dT"] / c["rho0"]
    return -c["gravity"] * alpha * gradient[None, :, None] * (z + c["H_m"]) / c["f"]


def native_depths(c, a, component=None):
    """Reference nodes for FD; actual evolving layer centroids for native layer models."""
    model = a["metadata"]["model"]
    size = a["eta"].shape[1] if component is None else a[component].shape[1]
    if model == "ocean-solver":
        return np.broadcast_to(-np.linspace(0, c["H_m"], c["nz"]), (len(a["time"]), size, c["nz"]))
    h = a["h"]
    eta = a["eta"]
    if component:
        field = h.reshape(len(a["time"]), c["nx"], c["ny"], c["nz"])
        surface = eta.reshape(len(a["time"]), c["nx"], c["ny"])
        if component == "u":
            h = 0.5 * (field + np.roll(field, 1, axis=1))
            eta = 0.5 * (surface + np.roll(surface, 1, axis=1))
        else:
            h = np.concatenate(
                (field[:, :, :1], 0.5 * (field[:, :, :-1] + field[:, :, 1:]), field[:, :, -1:]),
                axis=2,
            )
            eta = np.concatenate(
                (
                    surface[:, :, :1],
                    0.5 * (surface[:, :, :-1] + surface[:, :, 1:]),
                    surface[:, :, -1:],
                ),
                axis=2,
            )
        h = h.reshape(len(a["time"]), size, c["nz"])
        eta = eta.reshape(len(a["time"]), size)
    return eta[..., None] - np.cumsum(h, axis=-1) + h / 2


def reference_tracers(c, a, quadrature=32):
    shape = a["T"].shape
    if c["benchmark"] == "geostrophic-adjustment":
        return np.full(shape, c["T_C"]), np.full(shape, c["S_psu"])
    o = c["thermal_wind"]
    model = a["metadata"]["model"]
    alpha = -c["eos"]["drho_dT"] / c["rho0"]
    beta = c["eos"]["drho_dS"] / c["rho0"]
    z = native_depths(c, a)
    x = a["x_eta"]
    y = a["y_eta"]
    k = 2 * np.pi / c["Lx_m"]
    if model == "ocean-solver":
        delta, gradient = temperature_front(c, y)
        u = -c["gravity"] * alpha * gradient[None, :, None] * (z + c["H_m"]) / c["f"]
        spice = o["spice_T_C"] * np.cos(k * (x[None, :, None] - u * a["time"][:, None, None]))
        base = c["T_C"] + o["dT_dz_C_m"] * z + delta[None, :, None]
    else:
        nodes, weights = np.polynomial.legendre.leggauss(quadrature)
        delta, gradient = temperature_front(c, y[:, None] + c["Ly_m"] / c["ny"] * nodes / 2)
        shear = -c["gravity"] * alpha * gradient / c["f"]
        u = shear[None, :, None, :] * (z[..., None] + c["H_m"])
        kz = shear[None, :, None, :] * a["time"][:, None, None, None] * k
        values = np.cos(
            k * (x[None, :, None, None] - u * a["time"][:, None, None, None])
        ) * np.sinc(kz * a["h"][..., None] / (2 * np.pi))
        spice = o["spice_T_C"] * np.sinc(1 / c["nx"]) * np.sum(values * weights / 2, axis=-1)
        base = c["T_C"] + o["dT_dz_C_m"] * z + np.sum(delta * weights / 2, axis=-1)[None, :, None]
    return base + spice, c["S_psu"] + alpha / beta * spice


def validate(c, a, definition_validator=validate_contract):
    validate_native(c, a, definition_validator)
    width = 0 if a["metadata"]["model"] == "ocean-solver" else c["Ly_m"] / c["ny"]
    eta = reference_surface(c, a["time"][:1], a["y_eta"], width)[0]
    require(np.max(np.abs(a["eta"][0] - eta)) < 5e-11, "wrong initial channel sea level")
    T, S = reference_tracers(
        c,
        {
            **a,
            "time": a["time"][:1],
            "h": a["h"][:1],
            "eta": a["eta"][:1],
            "T": a["T"][:1],
            "S": a["S"][:1],
            "u": a["u"][:1],
            "v": a["v"][:1],
        },
    )
    require(
        np.max(np.abs(a["T"][0] - T[0])) < 5e-11 and np.max(np.abs(a["S"][0] - S[0])) < 5e-11,
        "wrong initial channel tracers",
    )
    for q in ("u", "v"):
        z = native_depths(c, a, q)[:1]
        expected = reference_velocity(c, a["time"][:1], a["y_" + q], z, q)[0]
        require(np.max(np.abs(a[q][0] - expected)) < 5e-11, "wrong initial channel velocity")


def score(c, a):
    if "wind_stress" in c:
        raise ValueError("forced trajectories require the dedicated response/budget evaluator")
    validate(c, a)
    width = 0 if a["metadata"]["model"] == "ocean-solver" else c["Ly_m"] / c["ny"]
    eta = reference_surface(c, a["time"], a["y_eta"], width)
    T, S = reference_tracers(c, a)
    u = reference_velocity(c, a["time"], a["y_u"], native_depths(c, a, "u"), "u")
    v = reference_velocity(c, a["time"], a["y_v"], native_depths(c, a, "v"), "v")
    scale_u = c["amplitude_m"] * np.sqrt(c["gravity"] / c["H_m"])
    if c["benchmark"] == "thermal-wind":
        o = c["thermal_wind"]
        alpha = -c["eos"]["drho_dT"] / c["rho0"]
        scale_u = (
            c["gravity"]
            * alpha
            * o["delta_T_C"]
            * np.pi
            * c["H_m"]
            / (2 * o["front_width_m"] * abs(c["f"]))
        )

    def rms(error, weights):
        spatial = np.sum(error**2 * weights, axis=tuple(range(1, error.ndim))) / np.sum(
            weights, axis=tuple(range(1, weights.ndim))
        )
        integral = np.sum(0.5 * (spatial[:-1] + spatial[1:]) * np.diff(a["time"]))
        return float(np.sqrt(integral / (a["time"][-1] - a["time"][0])))

    volume = a["h"] * a["area"][None, :, None]

    def inventory(field):
        return np.sum(field * volume, axis=(1, 2))

    heat, salt = inventory(a["T"]), inventory(a["S"])
    metrics = dict(
        eta_error=rms(a["eta"] - eta, np.broadcast_to(a["area"], eta.shape)) / c["amplitude_m"],
        u_error=rms(a["u"] - u, a["volume_u"]) / scale_u,
        v_error=rms(a["v"] - v, a["volume_v"]) / scale_u,
        T_error_C=rms(a["T"] - T, volume),
        S_error_psu=rms(a["S"] - S, volume),
        volume=float(np.max(np.abs(volume.sum(axis=(1, 2)) / volume[0].sum() - 1))),
        T_inventory=float(np.max(np.abs(heat / heat[0] - 1))),
        S_inventory=float(np.max(np.abs(salt / salt[0] - 1))),
    )
    failures = [
        key
        for key in ("eta_error", "u_error", "v_error")
        if metrics[key] > c["thresholds"]["error"]
    ]
    if metrics["volume"] > c["thresholds"]["volume"]:
        failures.append("volume")
    failures += [
        key
        for key in ("T_inventory", "S_inventory")
        if metrics[key] > c["thresholds"]["tracer_inventory"]
    ]
    failures += [key for key in ("T_error_C", "S_error_psu") if metrics[key] > c["thresholds"][key]]
    m = a["metadata"]
    return dict(
        schema=c["schema"],
        contract_sha256=digest(c),
        model=m["model"],
        metrics=metrics,
        failed_metrics=failures,
        engineering_screen_pass=not failures,
        industrial_qualified=False,
        performance_comparison=False,
        source_sha=m["source_sha"],
        executable_sha256=m["executable_sha256"],
        reference=c["reference"],
        units=dict(T_error_C="degC", S_error_psu="psu", other="relative"),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "output", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    c = json.loads(args.contract.read_text())
    result = score(c, load(args.output))
    result["output_sha256"] = sha256_file(args.output)
    write_json(args.report, result, create=True)
