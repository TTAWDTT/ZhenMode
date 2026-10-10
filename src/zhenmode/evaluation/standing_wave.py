"""Native-grid wave validation and unchanged v0 metrics; never runs a model."""

import argparse
import json
from pathlib import Path

import numpy as np

from zhenmode.benchmarks.standing_wave import (
    SCHEMA_V0,
    contract,
    digest,
    legacy_contract,
    method_contract,
    study_contract,
    validate_contract,
)
from zhenmode.evaluation.native_channel import load, require
from zhenmode.evaluation.native_channel import validate as validate_native
from zhenmode.provenance.sources import sha256_file


def exact(c, time, x, sampling="node", width=0.0):
    k = 2 * np.pi / c["Lx_m"]
    factor = 1.0 if sampling == "node" else np.sinc(k * np.asarray(width) / (2 * np.pi))
    if sampling not in {"node", "cell_mean"}:
        raise ValueError("unsupported sampling")
    phase = 2 * np.pi * np.asarray(time) / c["period_s"]
    eta = c["amplitude_m"] * np.cos(k * np.asarray(x))[None, :] * np.cos(phase)[:, None] * factor
    speed = c["amplitude_m"] * np.sqrt(c["gravity"] / c["H_m"])
    u = speed * np.sin(k * np.asarray(x))[None, :] * np.sin(phase)[:, None] * factor
    return eta, u


def validate(c, a):
    validate_native(c, a, validate_contract)
    t, m = a["time"], a["metadata"]
    dx = c["Lx_m"] / c["nx"]
    eta_ref, _ = exact(c, t, a["x_eta"], m["sampling_eta"], dx)
    require(
        np.max(np.abs(a["eta"][0] - eta_ref[0])) / c["amplitude_m"] <= 1e-10
        and np.max(np.abs(a["u"][0])) <= 1e-12
        and np.max(np.abs(a["v"][0])) <= 1e-12,
        "wrong initial wave/rest state",
    )



def score(c, a):
    validate(c, a)
    t, m = a["time"], a["metadata"]
    A, U = c["amplitude_m"], c["amplitude_m"] * np.sqrt(c["gravity"] / c["H_m"])
    eta_ref, _ = exact(c, t, a["x_eta"], m["sampling_eta"], c["Lx_m"] / c["nx"])
    _, u_ref = exact(c, t, a["x_u"], m["sampling_u"], a["width_u"])
    # Fixed reference geometry weights for error/modal fits; moving weights for KE.
    wu = a["volume_u"][0].sum(axis=1)
    ubar = (a["u"] * a["volume_u"]).sum(axis=2) / a["volume_u"].sum(axis=2)

    def rms(error, weights):
        return float(
            np.sqrt(
                np.sum(
                    0.5
                    * (
                        np.sum(error[:-1] ** 2 * weights, axis=1)
                        + np.sum(error[1:] ** 2 * weights, axis=1)
                    )
                    * np.diff(t)
                )
                / (weights.sum() * (t[-1] - t[0]))
            )
        )

    ce = eta_ref[0] / A
    su = u_ref[8] / U  # t=P/4: nonzero velocity basis, never a zero-crossing divisor.
    ae = np.sum(a["eta"] * ce * a["area"], axis=1) / (A * np.sum(ce**2 * a["area"]))
    au = np.sum(ubar * su * wu, axis=1) / (U * np.sum(su**2 * wu))
    require(np.sum(su**2 * wu) > 0.1 * wu.sum(), "native u samples do not resolve the wave")
    z = ae + 1j * au
    phase = np.angle(z * np.exp(-2j * np.pi * t / c["period_s"]))
    volume = (a["h"] * a["area"][None, :, None]).sum(axis=(1, 2))
    ke = 0.5 * c["rho0"] * sum((a["volume_" + q] * a[q] ** 2).sum(axis=(1, 2)) for q in ["u", "v"])
    pe = 0.5 * c["rho0"] * c["gravity"] * np.sum(a["eta"] ** 2 * a["area"], axis=1)
    energy = ke + pe
    metrics = dict(
        eta_error=rms(a["eta"] - eta_ref, a["area"]) / A,
        u_error=rms(ubar - u_ref, wu) / U,
        phase_rad=float(np.max(np.abs(phase))),
        amplitude=float(np.max(np.abs(np.abs(z) - 1))),
        energy=float(np.max(np.abs(energy / energy[0] - 1))),
        volume=float(np.max(np.abs(volume / volume[0] - 1))),
        T=float(np.max(np.abs(a["T"] - c["T_C"]))),
        S=float(np.max(np.abs(a["S"] - c["S_psu"]))),
        v_over_U=float(np.max(np.abs(a["v"])) / U),
    )
    bounds = dict(
        eta_error="error",
        u_error="error",
        phase_rad="phase_rad",
        amplitude="amplitude",
        energy="energy",
        volume="volume",
        T="tracer",
        S="tracer",
        v_over_U="v_over_U",
    )
    failures = [key for key, bound in bounds.items() if metrics[key] > c["thresholds"][bound]]
    return dict(
        schema=c["schema"],
        contract_sha256=digest(c),
        model=m["model"],
        source_sha=m["source_sha"],
        input_sha256=m["input_sha256"],
        config_sha256=m["config_sha256"],
        executable_sha256=m["executable_sha256"],
        metrics=metrics,
        failed_metrics=failures,
        engineering_screen_pass=not failures,
        industrial_qualified=False,
        equal_error_speedup_qualified=False,
        energy_units="J",
        initial_energy_J=float(energy[0]),
        resource_record={
            key: m[key]
            for key in [
                "initialization_s",
                "integration_s",
                "total_wall_s",
                "aggregate_peak_bytes",
                "cpu",
                "ranks",
            ]
        },
    )


def compare(c, reference, candidate, half=False):
    validate(c, reference)
    builder = legacy_contract if c["schema"] == SCHEMA_V0 else contract
    other = builder("medium", half=True) if half else c
    validate(other, candidate)
    for key in ["x_eta", "y_eta", "area", "x_u", "y_u", "width_u", "x_v", "y_v"]:
        require(np.array_equal(reference[key], candidate[key]), "paired native grids differ")
    require(
        reference["metadata"]["sampling_eta"] == candidate["metadata"]["sampling_eta"]
        and reference["metadata"]["sampling_u"] == candidate["metadata"]["sampling_u"],
        "paired sampling differs",
    )
    for key in [
        "model",
        "source_sha",
        "executable_sha256",
        "config_sha256",
        "recorded_numerics",
        "velocity_layout",
    ]:
        require(
            reference["metadata"][key] == candidate["metadata"][key], "paired model/config differs"
        )

    def mean_u(a):
        return (a["u"] * a["volume_u"]).sum(axis=2) / a["volume_u"].sum(axis=2)

    d_eta = reference["eta"] / c["amplitude_m"] - candidate["eta"] / other["amplitude_m"]
    d_u = (
        mean_u(reference) / c["amplitude_m"] - mean_u(candidate) / other["amplitude_m"]
    ) / np.sqrt(c["gravity"] / c["H_m"])
    if half:
        require(
            reference["metadata"]["trajectory"]["kind"]
            == candidate["metadata"]["trajectory"]["kind"]
            == "continuous",
            "half-amplitude requires continuous runs",
        )
        require(
            c["case"] == "medium" and c["amplitude_m"] == 0.01,
            "half-amplitude comparison requires main medium",
        )
        error = float(np.sqrt(np.mean(d_eta**2 + d_u**2) / 2))
        return dict(
            control="half_amplitude",
            error=error,
            engineering_screen_pass=error <= 0.002,
            industrial_qualified=False,
        )
    require(
        reference["metadata"]["trajectory"]["kind"] == "continuous"
        and candidate["metadata"]["trajectory"]["kind"] == "restart",
        "restart comparison requires independent restart receipt",
    )
    for key in ["volume_u", "volume_v"]:
        require(
            np.allclose(reference[key], candidate[key], rtol=1e-10, atol=0),
            "restart native quadrature differs",
        )
    error = max(float(np.max(np.abs(d_eta))), float(np.max(np.abs(d_u))))
    for key in ["u", "v"]:
        U = c["amplitude_m"] * np.sqrt(c["gravity"] / c["H_m"])
        error = max(error, float(np.max(np.abs(reference[key] - candidate[key])) / U))
    tracer = max(float(np.max(np.abs(reference[key] - candidate[key]))) for key in ["T", "S"])
    geometry = float(np.max(np.abs(reference["h"] - candidate["h"]))) / c["H_m"]
    error = max(error, geometry)
    return dict(
        control="restart",
        maximum_normalized_difference=error,
        maximum_tracer_difference=tracer,
        engineering_screen_pass=error <= 1e-10 and tracer <= 1e-10,
        industrial_qualified=False,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--case", choices=["coarse", "medium", "fine"], default="coarse")
    f.add_argument("--half", action="store_true")
    f.add_argument("--schema", choices=(SCHEMA_V0, "standing-wave-v1"), default="standing-wave-v1")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--study", nargs=2, type=int, metavar=("NX", "DT_SECONDS"))
    f.add_argument("--method", choices=("baseline", "symmetric-external-mode", "symmetric-closed-faces"), default="baseline")
    s = sub.add_parser("score")
    s.add_argument("--contract", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--report", type=Path, required=True)
    b = sub.add_parser("compare")
    b.add_argument("--contract", type=Path, required=True)
    b.add_argument("--reference", type=Path, required=True)
    b.add_argument("--candidate", type=Path, required=True)
    b.add_argument("--half", action="store_true")
    b.add_argument("--report", type=Path, required=True)
    args = p.parse_args(argv)
    if args.command == "freeze":
        if args.study and (args.half or args.schema == SCHEMA_V0):
            raise ValueError("space/time study requires the separate study protocol")
        if args.schema == SCHEMA_V0 and args.method != "baseline":
            raise ValueError("v0 cannot declare the new external method")
        builder = legacy_contract if args.schema == SCHEMA_V0 else (
            contract if args.method == "baseline" else method_contract
        )
        frozen = study_contract(*args.study, method=args.method) if args.study else (
            builder(args.case, args.half) if args.method == "baseline"
            else method_contract(args.case, args.half, args.method)
        )
        with args.out.open("x", encoding="utf8") as stream:
            stream.write(
                json.dumps(
                    frozen,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
        return
    c = json.loads(args.contract.read_text())
    result = (
        score(c, load(args.output))
        if args.command == "score"
        else compare(c, load(args.reference), load(args.candidate), args.half)
    )
    files = (
        {"output_sha256": args.output}
        if args.command == "score"
        else {"reference_output_sha256": args.reference, "candidate_output_sha256": args.candidate}
    )
    for key, path in files.items():
        result[key] = sha256_file(path)
    with args.report.open("x", encoding="utf8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["engineering_screen_pass"]:
        raise SystemExit(1)
