"""Separate analytic oracle on evolving native geometry for the frozen internal mode."""

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np

from zhenmode.benchmarks.internal_wave import validate_contract
from zhenmode.benchmarks.standing_wave import digest
from zhenmode.evaluation.native_channel import load_snapshot, require
from zhenmode.evaluation.native_channel import validate as validate_native


def geometry(c, a, component=None):
    """Node positions or actual scalar/face layer centroids and complete layer widths."""
    nt = len(a["time"])
    size = a["eta"].shape[1] if component is None else a[component].shape[1]
    if a["metadata"]["model"] == "ocean-solver":
        return np.broadcast_to(-np.linspace(0, c["H_m"], c["nz"]), (nt, size, c["nz"])), 0.0
    h, eta = a["h"], a["eta"]
    if component:
        h = h.reshape(nt, c["nx"], c["ny"], c["nz"])
        eta = eta.reshape(nt, c["nx"], c["ny"])
        if component == "u":
            h = (h + np.roll(h, 1, axis=1)) / 2
            eta = (eta + np.roll(eta, 1, axis=1)) / 2
        else:
            h = np.concatenate((h[:, :, :1], (h[:, :, :-1] + h[:, :, 1:]) / 2, h[:, :, -1:]), axis=2)
            eta = np.concatenate((eta[:, :, :1], (eta[:, :, :-1] + eta[:, :, 1:]) / 2, eta[:, :, -1:]), axis=2)
        h, eta = h.reshape(nt, size, c["nz"]), eta.reshape(nt, size)
    upper = eta[..., None] - np.concatenate((np.zeros_like(h[..., :1]), np.cumsum(h[..., :-1], axis=-1)), axis=-1)
    return upper - h / 2, h


def reference(c, a):
    """Closed form oracle; never import or call the initialization producer."""
    mode = c["internal_wave"]
    m = mode["q"] / c["H_m"]
    k = np.pi / c["Ly_m"]
    omega = np.sqrt(c["f"]**2 + mode["N2_s_2"] * (k / m)**2)
    A, gamma = c["amplitude_m"], mode["dT_dz_C_m"]
    phase = omega * a["time"][:, None, None]
    horizontal_factor = 1.0 if a["metadata"]["model"] == "ocean-solver" else np.sinc(1 / (2 * c["ny"]))
    eta = A * np.sin(mode["q"]) * np.cos(k * a["y_eta"])[None, :] * np.cos(phase[..., 0]) * horizontal_factor
    z, h = geometry(c, a)
    mean_sin = np.sin(m * (z + c["H_m"])) * np.sinc(m * h / (2 * np.pi))
    T = c["T_C"] + gamma * z - gamma * A * mean_sin * np.cos(k * a["y_eta"])[None, :, None] * np.cos(phase) * horizontal_factor
    result = dict(eta=eta, T=T, S=np.full(T.shape, c["S_psu"]))
    for q in ("u", "v"):
        z, h = geometry(c, a, q)
        shape = np.cos(m * (z + c["H_m"])) * np.sinc(m * h / (2 * np.pi)) * np.sin(k * a["y_" + q])[None, :, None]
        result[q] = shape * A * m / k * (-c["f"] * np.cos(phase) if q == "u" else omega * np.sin(phase))
    return result


def validate(c, a):
    validate_native(c, a, validate_contract)
    first = {key: value[:1] if key in ("time", "eta", "h", "T", "S", "u", "v") else value for key, value in a.items()}
    expected = reference(c, first)
    for q, field in expected.items():
        require(np.max(np.abs(a[q][0] - field[0])) < 5e-11, "wrong initial internal-wave " + q)


def score(c, a):
    validate(c, a)
    r = reference(c, a)
    A = c["amplitude_m"] or 0.01
    mode = c["internal_wave"]
    m, k = mode["q"] / c["H_m"], np.pi / c["Ly_m"]
    omega = np.sqrt(c["f"]**2 + mode["N2_s_2"] * (k / m)**2)

    def rms(error, weight):
        axes = tuple(range(1, error.ndim))
        variance = np.sum(weight * error**2, axis=axes) / np.sum(weight, axis=axes)
        return float(np.sqrt(np.sum((variance[1:] + variance[:-1]) * np.diff(a["time"]) / 2) / c["duration_s"]))

    volume = a["h"] * a["area"][None, :, None]
    totals = {q: np.sum(a[q] * volume, axis=(1, 2)) for q in ("T", "S")}
    metrics = dict(
        eta_error=rms(a["eta"] - r["eta"], np.broadcast_to(a["area"], a["eta"].shape)) / abs(A * np.sin(mode["q"])),
        u_error=rms(a["u"] - r["u"], a["volume_u"]) / (A * omega * m / k),
        v_error=rms(a["v"] - r["v"], a["volume_v"]) / (A * omega * m / k),
        T_error_C=rms(a["T"] - r["T"], volume), S_error_psu=rms(a["S"] - r["S"], volume),
        volume=float(np.max(np.abs(volume.sum(axis=(1, 2)) / volume[0].sum() - 1))),
        **{q + "_inventory": float(np.max(np.abs(totals[q] / totals[q][0] - 1))) for q in totals},
    )
    limits = {q: c["thresholds"]["error"] for q in ("eta_error", "u_error", "v_error")}
    limits.update({q: c["thresholds"][q] for q in ("T_error_C", "S_error_psu", "volume")})
    limits.update({q: c["thresholds"]["tracer_inventory"] for q in ("T_inventory", "S_inventory")})
    failures = [q for q in metrics if metrics[q] > limits[q]]
    null_errors = {}
    if c["control"] == "null":
        for q in ("eta", "u", "v", "T", "S", "h"):
            null_errors[q] = float(np.max(np.abs(a[q] - a[q][:1])))
            if null_errors[q] > c["thresholds"]["null_absolute"]:
                failures.append("null_" + q)
    keys = ("eta_error", "u_error", "v_error", "T_error_C", "S_error_psu")
    ratios = {q: metrics[q] / limits[q] for q in keys}
    return dict(schema=c["schema"], contract_sha256=digest(c), model=a["metadata"]["model"],
                metrics=metrics, failed_metrics=failures, engineering_screen_pass=not failures,
                normalized_error_ratios=ratios, composite=max(ratios.values()),
                null_absolute_errors=null_errors, industrial_qualified=False, performance_comparison=False,
                source_sha=a["metadata"]["source_sha"], executable_sha256=a["metadata"]["executable_sha256"],
                reference=c["reference"], units=dict(T_error_C="degC", S_error_psu="absolute psu", other="relative"))


def _json_report(result):
    """Represent overflow explicitly without changing the scientific scorer."""
    nonfinite = {}

    def encode(value, path=""):
        if isinstance(value, float) and not math.isfinite(value):
            nonfinite[path] = "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
            return None
        if isinstance(value, dict):
            return {key: encode(item, path + "/" + key) for key, item in value.items()}
        if isinstance(value, list):
            return [encode(item, path + "/" + str(i)) for i, item in enumerate(value)]
        return value

    report = encode(result)
    if nonfinite:
        report["non_finite_derived_values"] = nonfinite
        report["engineering_screen_pass"] = False
        report["failed_metrics"].append("non_finite_derived_values")
    return report


def main(argv=None):
    """Score existing contract/output files without running the model."""
    parser = argparse.ArgumentParser(
        prog="zhenmode evaluate internal-wave", description=__doc__
    )
    for name in ("contract", "output", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        c = json.loads(args.contract.read_text(encoding="utf-8"))
        require(isinstance(c, dict), "contract must be a JSON object")
        arrays, output_sha256 = load_snapshot(args.output)
        result = _json_report(score(c, arrays))
        result["output_sha256"] = output_sha256
        # Complete serialization before creating a report, so encoding failures
        # cannot leave a truncated evidence file that prevents retries.
        encoded = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    except (ValueError, TypeError, OSError, KeyError, EOFError, RuntimeError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    return 0
