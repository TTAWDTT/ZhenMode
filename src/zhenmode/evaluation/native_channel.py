"""Shared native flat-channel output reader and geometry/provenance checks."""

import json
import re
import zipfile
from pathlib import Path

import numpy as np

from zhenmode.benchmarks.standing_wave import MOM6, OCEANANIGANS, SCHEMA_V0, digest


def duration(c):
    return c["duration_s"] if "duration_s" in c else c["period_s"]


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


def validate(c, a, definition_validator):
    definition_validator(c)
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
        m["schema"] == c["schema"] and m["contract_sha256"] == digest(c),
        "wrong schema/contract hash",
    )
    require(m["model"] in c["native_sampling"], "unsupported model")
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
    pinned = {"MOM6": MOM6, "Oceananigans": OCEANANIGANS}
    require(
        m["model"] not in pinned or m["source_sha"] == pinned[m["model"]],
        "external model source is not pinned",
    )
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
        and m["steps"] == int(duration(c) / c["dt"]),
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
    expected_sampling = c["native_sampling"][m["model"]]
    require(
        m["sampling_eta"] == expected_sampling["eta"]
        and m["sampling_u"] == expected_sampling["u"]
        and m["velocity_layout"] == expected_sampling["layout"],
        "model sampling/layout differs from its frozen contract",
    )
    if m["velocity_layout"] == "cgrid":
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
    option_key = {
        "ocean-solver": "ocean_options",
        "MOM6": "mom_time_options",
        "Oceananigans": "oceananigans_options",
    }[m["model"]]
    expected_options = c[option_key]
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
    t = np.arange(0, duration(c) + 1, c["output_s"])
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
    if m["model"] == "Oceananigans":
        h0 *= 1 + a["eta"][0, :, None] / c["H_m"]
    else:
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
        # v1 accepts the same metre-level roundoff already allowed by its
        # independent native-coordinate check above. Preserve exact v0 bounds.
        coordinate_roundoff = 0.0 if c["schema"] == SCHEMA_V0 else 1e-9
        require(
            np.all(
                (a["x_" + component] >= -coordinate_roundoff)
                & (a["x_" + component] <= c["Lx_m"] + coordinate_roundoff)
            ),
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
