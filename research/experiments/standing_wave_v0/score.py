"""Frozen engineering screening for full-model f=0 standing waves; no solver."""

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np

SCHEMA = "standing-wave-v0"
MOM6 = "f49a00096df607b48354603e2398e14e189fd62e"
FIELDS = {
    "metadata",
    "time",
    "x_eta",
    "y_eta",
    "area",
    "eta",
    "h",
    "T",
    "S",
    "x_u",
    "y_u",
    "width_u",
    "u",
    "volume_u",
    "x_v",
    "y_v",
    "v",
    "volume_v",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def contract(case="coarse", half=False):
    nx, dt, error = {
        "coarse": (64, 100.0, 0.10),
        "medium": (128, 50.0, 0.05),
        "fine": (256, 25.0, 0.02),
    }[case]
    if half and case != "medium":
        raise ValueError("half-amplitude control is frozen at medium only")
    return dict(
        schema=SCHEMA,
        case=case,
        nx=nx,
        ny=8,
        nz=4,
        dt=dt,
        period_s=32000.0,
        output_s=1000.0,
        Lx_m=float(32000 * np.sqrt(981.0)),
        Ly_m=100000.0,
        H_m=100.0,
        gravity=9.81,
        rho0=1025.0,
        f=0.0,
        T_C=15.0,
        S_psu=35.0,
        amplitude_m=0.005 if half else 0.01,
        initial_h_m=[100 / 6, 100 / 3, 100 / 3, 100 / 6],
        eos=dict(name="linear", drho_dT=-0.205, drho_dS=0.779, drho_dp=0.0),
        boundary=dict(x="periodic", y="closed", bottom="flat"),
        native_sampling={
            "ocean-solver": {"eta": "node", "u": "node", "layout": "collocated"},
            "MOM6": {"eta": "cell_mean", "u": "node", "layout": "cgrid"},
        },
        ocean_options=dict(
            column_geometry="nodal_dual_v1",
            process_time_scheme="legacy",
            mode_split=False,
            conservative_kv=True,
            localize_conv=True,
            project_adv_vel=False,
            monotone_adv=False,
            fct_adv=False,
            polar_cap_rows=0,
            polar_cap_taper=0,
        ),
        mom_time_options=dict(
            SPLIT=True, SPLIT_RK2B=False, DT=dt, DT_THERM=dt, DT_FORCING=dt, DTBT=dt
        ),
        thresholds=dict(
            error=error,
            phase_rad=0.05,
            amplitude=0.02,
            energy=0.02,
            volume=1e-10,
            tracer=1e-10,
            v_over_U=1e-8,
            restart=1e-10,
        ),
        explicitly_zero=[
            "wind",
            "heat_salt_sources",
            "restoring",
            "bottom_drag",
            "explicit_viscosity",
            "explicit_diffusion",
            "convection_enhancement",
            "GM_Redi",
            "ice",
            "sponge",
            "polar_filter",
        ],
        qualification="engineering_screen_only",
        full_dynamics=True,
        budget=dict(cpu=1, ranks=1, aggregate_peak_bytes=4 * 1024**3, coarse_wall_seconds=600),
    )


def validate_contract(c):
    if not isinstance(c, dict) or c != contract(c.get("case"), c.get("amplitude_m") == 0.005):
        raise ValueError("contract differs from frozen v0")


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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    """Small declared NPZ adapter output; allow_pickle is always disabled."""
    require(Path(path).stat().st_size <= 128 * 1024**2, "output exceeds 128 MiB file budget")
    with zipfile.ZipFile(path) as z:
        require(
            sum(info.file_size for info in z.infolist()) <= 128 * 1024**2,
            "decoded output exceeds 128 MiB budget",
        )
    with np.load(path, allow_pickle=False) as a:
        require(set(a.files) == FIELDS, "missing or unknown output arrays")
        result = {name: a[name] for name in a.files}
    require(
        result["metadata"].ndim == 0 and result["metadata"].dtype.kind == "U",
        "metadata must be JSON text",
    )
    result["metadata"] = json.loads(str(result["metadata"]))
    return result


def validate(c, a):
    validate_contract(c)
    require(set(a) == FIELDS, "missing or unknown output arrays")
    m = a["metadata"]
    required = {
        "schema",
        "contract_sha256",
        "model",
        "source_sha",
        "executable_sha256",
        "input_sha256",
        "config_sha256",
        "coordinate_system",
        "units",
        "sampling_eta",
        "snapshot_kind",
        "sampling_u",
        "velocity_layout",
        "dt_s",
        "steps",
        "full_dynamics",
        "zero_processes",
        "recorded_numerics",
        "initialization_s",
        "integration_s",
        "total_wall_s",
        "aggregate_peak_bytes",
        "cpu",
        "ranks",
        "trajectory",
        "g",
        "rho0",
        "H",
        "Lx",
        "Ly",
        "f",
        "eos",
        "boundary",
    }
    require(isinstance(m, dict) and set(m) == required, "metadata fields differ from contract")
    require(
        m["schema"] == SCHEMA and m["contract_sha256"] == digest(c), "wrong schema/contract hash"
    )
    require(m["model"] in {"ocean-solver", "MOM6"}, "unsupported model")
    for name, size in [
        ("source_sha", 40),
        ("executable_sha256", 64),
        ("input_sha256", 64),
        ("config_sha256", 64),
    ]:
        require(
            isinstance(m[name], str)
            and re.fullmatch("[0-9a-f]{" + str(size) + "}", m[name]) is not None,
            "invalid " + name,
        )
    require(m["model"] != "MOM6" or m["source_sha"] == MOM6, "MOM6 source is not pinned")
    require(m["coordinate_system"] == "cartesian_m", "wrong coordinate system")
    require(
        m["units"]
        == dict(
            time="s", x="m", eta="m", h="m", u="m/s", volume="m3", area="m2", T="degC", S="psu"
        ),
        "wrong units",
    )
    for name, cname in [
        ("g", "gravity"),
        ("rho0", "rho0"),
        ("H", "H_m"),
        ("Lx", "Lx_m"),
        ("Ly", "Ly_m"),
        ("f", "f"),
        ("eos", "eos"),
        ("boundary", "boundary"),
    ]:
        require(m[name] == c[cname], "wrong physical metadata: " + name)
    require(
        m["dt_s"] == c["dt"]
        and type(m["steps"]) is int
        and m["steps"] == int(c["period_s"] / c["dt"]),
        "wrong timestep/step count",
    )
    require(
        m["full_dynamics"] is True and sorted(m["zero_processes"]) == sorted(c["explicitly_zero"]),
        "incomplete dynamics/disabled-process declaration",
    )
    require(
        isinstance(m["recorded_numerics"], dict)
        and all(
            isinstance(m["recorded_numerics"].get(key), str) and m["recorded_numerics"][key].strip()
            for key in ["time_scheme", "transport", "filters", "vertical_coordinate", "substeps"]
        ),
        "missing numerical-process records",
    )
    require(m["snapshot_kind"] == "instantaneous", "time-averaged output is not a snapshot")
    if m["model"] == "MOM6":
        require(
            m["sampling_eta"] == "cell_mean"
            and m["sampling_u"] == "node"
            and m["velocity_layout"] == "cgrid",
            "MOM6 native sampling/layout mismatch",
        )
    else:
        require(
            m["sampling_eta"] == m["sampling_u"] == "node" and m["velocity_layout"] == "collocated",
            "ocean native sampling/layout mismatch",
        )
    expected_options = c["ocean_options"] if m["model"] == "ocean-solver" else c["mom_time_options"]
    require(
        m["recorded_numerics"].get("resolved_options") == expected_options,
        "time/coordinate options differ from frozen configuration",
    )
    tr = m["trajectory"]
    require(
        isinstance(tr, dict) and tr.get("kind") in {"continuous", "restart"},
        "missing trajectory receipt",
    )
    if tr["kind"] == "continuous":
        require(tr == {"kind": "continuous", "processes": 1}, "invalid continuous receipt")
    else:
        require(
            set(tr) == {"kind", "processes", "split_s", "checkpoint_sha256", "full_native_state"}
            and tr["processes"] == 2
            and tr["split_s"] == 16000.0
            and tr["full_native_state"] is True
            and isinstance(tr["checkpoint_sha256"], str)
            and re.fullmatch("[0-9a-f]{64}", tr["checkpoint_sha256"]) is not None,
            "invalid cross-process restart receipt",
        )
    for name in ["initialization_s", "integration_s", "total_wall_s", "aggregate_peak_bytes"]:
        require(
            type(m[name]) in {int, float} and np.isfinite(m[name]) and m[name] >= 0,
            "invalid resource record",
        )
    require(
        m["integration_s"] > 0 and m["total_wall_s"] >= m["initialization_s"] + m["integration_s"],
        "inconsistent timings",
    )
    require(
        type(m["cpu"]) is int and type(m["ranks"]) is int and m["cpu"] == m["ranks"] == 1,
        "wrong CPU/rank budget",
    )
    require(
        0 < m["aggregate_peak_bytes"] <= c["budget"]["aggregate_peak_bytes"],
        "missing/exceeded aggregate memory budget",
    )
    if c["case"] == "coarse":
        require(m["total_wall_s"] <= 600, "coarse run exceeds 10-minute budget")
    for name in FIELDS - {"metadata"}:
        require(
            isinstance(a[name], np.ndarray)
            and a[name].dtype.kind == "f"
            and a[name].dtype.itemsize == 8
            and np.all(np.isfinite(a[name])),
            "non-finite/non-float64 array: " + name,
        )
    t = np.arange(0, c["period_s"] + 1, c["output_s"])
    require(np.array_equal(a["time"], t), "wrong time or incomplete output")
    nt, ne, nz = len(t), c["nx"] * c["ny"], c["nz"]
    for name in ["x_eta", "y_eta", "area"]:
        require(a[name].shape == (ne,), "wrong grid shape")
    dx, dy = c["Lx_m"] / c["nx"], c["Ly_m"] / c["ny"]
    xy = np.stack([a["x_eta"], a["y_eta"]], axis=1)
    expected = np.stack(
        np.meshgrid(
            (np.arange(c["nx"]) + 0.5) * dx, (np.arange(c["ny"]) + 0.5) * dy, indexing="ij"
        ),
        axis=-1,
    ).reshape(ne, 2)
    require(
        np.allclose(xy, expected, rtol=0, atol=1e-9)
        and np.allclose(a["area"], dx * dy, rtol=1e-13, atol=0),
        "wrong cell locations/metrics",
    )
    require(a["eta"].shape == (nt, ne), "wrong eta shape")
    for name in ["h", "T", "S"]:
        require(a[name].shape == (nt, ne, nz), "wrong thickness/tracer shape")
    require(np.all(a["h"] > 0), "non-positive thickness")
    require(
        np.max(np.abs(a["h"].sum(axis=2) - c["H_m"] - a["eta"])) <= 1e-10,
        "eta/thickness geometry mismatch",
    )
    h0 = np.broadcast_to(np.asarray(c["initial_h_m"]), (ne, nz)).copy()
    h0[:, 0] += a["eta"][0]
    require(np.max(np.abs(a["h"][0] - h0)) <= 1e-10, "wrong initial vertical geometry")
    layout = m["velocity_layout"]
    require(layout in {"collocated", "cgrid"}, "unsupported native velocity layout")
    hh = a["h"].reshape(nt, c["nx"], c["ny"], nz)
    native = {}
    if layout == "collocated":
        native = {q: (expected, a["h"] * a["area"][None, :, None]) for q in ["u", "v"]}
    else:
        ux, uy = np.meshgrid(
            np.arange(c["nx"]) * dx, (np.arange(c["ny"]) + 0.5) * dy, indexing="ij"
        )
        vx, vy = np.meshgrid(
            (np.arange(c["nx"]) + 0.5) * dx, np.arange(c["ny"] + 1) * dy, indexing="ij"
        )
        vu = 0.5 * (hh + np.roll(hh, 1, axis=1)) * dx * dy
        vv = (
            np.concatenate(
                (0.5 * hh[:, :, :1], 0.5 * (hh[:, :, :-1] + hh[:, :, 1:]), 0.5 * hh[:, :, -1:]),
                axis=2,
            )
            * dx
            * dy
        )
        native = dict(
            u=(np.stack([ux, uy], axis=-1).reshape(-1, 2), vu.reshape(nt, -1, nz)),
            v=(np.stack([vx, vy], axis=-1).reshape(-1, 2), vv.reshape(nt, -1, nz)),
        )
    for component in ["u", "v"]:
        positions, quadrature = native[component]
        require(
            a["y_" + component].shape == a["x_" + component].shape,
            "wrong native velocity y coordinates",
        )
        actual = np.stack([a["x_" + component], a["y_" + component]], axis=-1)
        require(
            actual.shape == positions.shape and np.allclose(actual, positions, rtol=0, atol=1e-9),
            "wrong native velocity positions",
        )
        require(
            a["volume_" + component].shape == quadrature.shape
            and np.allclose(a["volume_" + component], quadrature, rtol=1e-10, atol=0),
            "native kinetic quadrature differs from declared geometry",
        )
        count = a["x_" + component].size
        require(
            c["nx"] <= count <= 2 * ne and a["x_" + component].shape == (count,),
            "wrong native velocity grid",
        )
        require(
            np.all((a["x_" + component] >= 0) & (a["x_" + component] <= c["Lx_m"])),
            "velocity coordinates outside domain",
        )
        require(
            a[component].shape == (nt, count, nz)
            and a["volume_" + component].shape == (nt, count, nz),
            "wrong native velocity/dual-volume shape",
        )
        require(np.all(a["volume_" + component] > 0), "non-positive native kinetic quadrature")
        require(
            np.max(
                np.abs(
                    a["volume_" + component].sum(axis=(1, 2))
                    / ((a["h"] * a["area"][None, :, None]).sum(axis=(1, 2)))
                    - 1
                )
            )
            <= 1e-10,
            "native kinetic quadrature has wrong total volume",
        )
    require(a["width_u"].shape == a["x_u"].shape, "wrong native u sampling widths")
    require(
        m["sampling_eta"] in {"node", "cell_mean"} and m["sampling_u"] in {"node", "cell_mean"},
        "unsupported sampling",
    )
    require(
        np.all(a["width_u"] == (0.0 if m["sampling_u"] == "node" else dx)),
        "wrong native u sampling width",
    )
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
        schema=SCHEMA,
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
    other = contract("medium", half=True) if half else c
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--case", choices=["coarse", "medium", "fine"], default="coarse")
    f.add_argument("--half", action="store_true")
    f.add_argument("--out", type=Path, required=True)
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
    args = p.parse_args()
    if args.command == "freeze":
        args.out.write_text(
            json.dumps(contract(args.case, args.half), indent=2, allow_nan=False) + "\n"
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
        result[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["engineering_screen_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
