"""Bounded synthetic component report, always unqualified for ocean repair."""
import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import component as m
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    config = {"shape": [2, 3], "area_m2": 1.0, "bottom_z_m": -20., "band_bottom_z_m": -10.,
              "moving_top_fraction": .25, "dt_s": 1., "q_left_to_right_m3_per_s": [.25, .35, 0.],
              "steps_down": 6, "steps_return": 6, "cfl_limit": m.CFL_LIMIT,
              "field_semantics": "finite_volume_cell_means_T_S_u_v",
              "sources": "zero_in_cycle; nonzero_extensive_sources_tested_separately",
              "initial_h": m.initial().h.tolist(), "initial_means": m.means(m.initial()).tolist()}
    config_path = args.output / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    report = {"base": "82e1ca4a2158d4c0ed20f91c04e80cdf43f8a36b",
              "scope": "synthetic_prescribed_transport_FV_component",
              "real_inputs_used": False, "coupled_ocean_steps": 0, "qualification_passed": False,
              "pressure_reconstruction_qualified": False, "results": {}}
    for scheme in ("merge", "moving"):
        initial = m.initial()
        state, ok, _ = m.remap(initial, scheme)
        if not ok:
            raise RuntimeError("conversion failed")
        converted = state.copy()
        path = []
        for index in range(12):
            q = np.array([.25, .35, 0.]) * (1 if state.step < 6 else -1)
            state, ok, reason = m.advance(state, q, np.zeros((2, 3, 4)))
            if not ok:
                raise RuntimeError(reason)
            path.append({"eta": state.eta.tolist(), "minimum_active_h": float(state.h[state.h > 0].min())})
        mass = m.RHO0 * initial.h[:, :2]
        shear = m.means(initial)[:, 0, 2:] - m.means(initial)[:, 1, 2:]
        merge_loss = np.sum(mass[:, 0] * mass[:, 1] / (2 * mass.sum(axis=1)) * np.sum(shear**2, axis=1))
        report["results"][scheme] = {
            "eta_path": path, "water_residual": float(state.h.sum() - initial.h.sum()),
            "content_residual_T_S_u_v": (state.n.sum(axis=(0, 1)) - initial.n.sum(axis=(0, 1))).tolist(),
            "conversion_energy_loss_J": m.energy(initial) - m.energy(converted),
            "merge_formula_J": float(merge_loss) if scheme == "merge" else None,
            "cycle_energy_loss_J": m.energy(converted) - m.energy(state),
            "initial_temperature_variance": m.variance(initial, 0),
            "converted_temperature_variance": m.variance(converted, 0),
            "final_temperature_variance": m.variance(state, 0),
            "cycle_content_maxabs": float(np.max(np.abs(state.n - converted.n))),
            "initial_active_means": m.means(converted)[converted.h > 0].tolist(),
            "final_active_means": m.means(state)[state.h > 0].tolist()}
    report["pressure_counterexample_m_per_s2"] = m.pressure_counterexample()
    root = Path(__file__).resolve().parents[3]
    paths = [Path(__file__).resolve(), Path(__file__).with_name("component.py"),
             root / "tests/test_top_band_controls.py", config_path]
    report["sha256"] = {path.name: digest(path) for path in paths}
    report["cost"] = {"wall_s": time.perf_counter() - began,
                      "peak_rss_KiB_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "python": platform.python_version(), "numpy": np.__version__,
                      "device": "CPU", "billing_cost": "not_visible"}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"qualification_passed": False,
                      "pressure_counterexample_m_per_s2": report["pressure_counterexample_m_per_s2"],
                      "cost": report["cost"]}))


if __name__ == "__main__":
    main()
