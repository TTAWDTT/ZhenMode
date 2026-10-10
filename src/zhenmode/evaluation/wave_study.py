"""Separate spatial/time controls and measured modal trends; never runs a model."""

import argparse
import json
from pathlib import Path

import numpy as np

from zhenmode.benchmarks.standing_wave import METHOD_STUDY_SCHEMA, STUDY_PAIRS, STUDY_SCHEMA, digest
from zhenmode.evaluation.standing_wave import exact, load, require, score, validate
from zhenmode.execution.runs import write_json
from zhenmode.provenance.sources import sha256_file


def modal_trends(c, a):
    """Fit phase and log-amplitude vs time, retaining residuals rather than claiming an order."""
    validate(c, a)
    time, m = a["time"], a["metadata"]
    amplitude = c["amplitude_m"]
    speed = amplitude * np.sqrt(c["gravity"] / c["H_m"])
    eta_basis = exact(c, time[:1], a["x_eta"], m["sampling_eta"], c["Lx_m"] / c["nx"])[0][0]
    velocity_basis = exact(
        c, np.array([c["period_s"] / 4]), a["x_u"], m["sampling_u"], a["width_u"]
    )[1][0]
    weights = a["volume_u"][0].sum(axis=1)
    velocity = (a["u"] * a["volume_u"]).sum(axis=2) / a["volume_u"].sum(axis=2)
    eta_mode = np.sum(a["eta"] * eta_basis * a["area"], axis=1) / np.sum(eta_basis**2 * a["area"])
    velocity_mode = np.sum(velocity * velocity_basis * weights, axis=1) / np.sum(
        velocity_basis**2 * weights
    )
    mode = eta_mode + 1j * velocity_mode
    require(np.all(np.abs(mode) > 0.1), "wave mode too small for phase/log-amplitude fit")
    phase = np.unwrap(np.angle(mode))
    require(np.all(np.abs(np.diff(phase)) < np.pi / 2), "phase sampling is too sparse")
    x = time / c["period_s"]
    design = np.column_stack((np.ones_like(x), x))
    phase_fit = np.linalg.lstsq(design, phase, rcond=None)[0]
    amplitude_fit = np.linalg.lstsq(design, np.log(np.abs(mode)), rcond=None)[0]
    return {
        "phase_rad": phase.tolist(),
        "modal_amplitude": np.abs(mode).tolist(),
        "relative_frequency_bias": float(phase_fit[1] / (2 * np.pi) - 1),
        "log_amplitude_change_per_period": float(amplitude_fit[1]),
        "phase_fit_max_residual_rad": float(np.max(np.abs(phase - design @ phase_fit))),
        "log_amplitude_fit_max_residual": float(
            np.max(np.abs(np.log(np.abs(mode)) - design @ amplitude_fit))
        ),
        "normalization": {"eta_m": amplitude, "velocity_m_s": speed},
    }


MODELS = ("zhenmode", "mom6", "oceananigans")


def analyze(directories, output, models=MODELS):
    """Rescore all five pairs for explicitly selected models; default remains three."""
    require(
        bool(models) and len(models) == len(set(models)) and set(models) <= set(MODELS),
        "select distinct supported study models",
    )
    rows, identities = [], []
    programs, package = {}, None
    definition = None
    seen = set()
    for directory in map(Path, directories):
        c = json.loads((directory / "contract.json").read_text())
        require(c["schema"] in (STUDY_SCHEMA, METHOD_STUDY_SCHEMA), "requires the separate space/time protocol")
        current = (c["schema"], c.get("method", "baseline"))
        if definition is None:
            definition = current
        require(current == definition, "study methods/protocols differ")
        pair = (c["nx"], int(c["dt"]))
        require(pair not in seen, "duplicate study configuration")
        seen.add(pair)
        receipt = json.loads((directory / "comparison.json").read_text())
        require(
            receipt["status"] == "completed"
            and set(receipt["models"]) == set(models),
            "incomplete selected-model study",
        )
        if package is None:
            package = receipt["package_source_sha256"]
        require(package == receipt["package_source_sha256"], "study package sources differ")
        for model in models:
            path = directory / model / "output.npz"
            a = load(path)
            report = score(c, a)
            require(
                report["model"]
                == {"zhenmode": "ocean-solver", "mom6": "MOM6", "oceananigans": "Oceananigans"}[
                    model
                ],
                "model directory/identity differs",
            )
            identity = (report["source_sha"], report["executable_sha256"])
            if model not in programs:
                programs[model] = identity
            require(programs[model] == identity, "study source/program identity differs")
            saved = receipt["models"][model]
            require(
                saved["output_sha256"] == sha256_file(path)
                and saved["metrics"] == report["metrics"],
                "study output/score changed",
            )
            rows.append(
                dict(
                    nx=pair[0],
                    dt_s=pair[1],
                    model=model,
                    metrics=report["metrics"],
                    trends=modal_trends(c, a),
                    failed_metrics=report["failed_metrics"],
                )
            )
        identities.append(
            dict(
                path=str(directory.resolve()),
                contract_sha256=digest(c),
                comparison_sha256=sha256_file(directory / "comparison.json"),
            )
        )
    require(seen == set(STUDY_PAIRS), "study requires exactly the five frozen grid/time pairs")
    result = dict(
        schema="standing-wave-study-report-v1",
        configurations=5,
        native_runs=5 * len(models),
        rows=rows,
        inputs=identities,
        spatial_axis={"dt_s": 25, "nx": [64, 128, 256]},
        time_axis={"nx": 256, "dt_s": [100, 50, 25]},
        interpretation="Modal linear trends include fit residuals; no whole-model order or independent spatial/time error decomposition is claimed.",
    )
    if definition != (STUDY_SCHEMA, "baseline") or tuple(models) != MODELS:
        result.update(schema="standing-wave-study-report-v2", models=list(models), method=definition[1])
    write_json(output, result, create=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=5, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = parser.parse_args(argv)
    analyze(args.runs, args.output, tuple(args.models))


if __name__ == "__main__":
    main()
