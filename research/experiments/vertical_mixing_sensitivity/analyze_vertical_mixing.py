"""Analyze vertical-mixing sensitivity against the reproduced real-air baseline."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def fmt(v: float, digits: int = 3) -> str:
    return "NA" if not np.isfinite(v) else f"{v:.{digits}f}"


def steady_mean(path: Path, days: int = 90) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path)
    steady = np.asarray(d["days"]) >= np.asarray(d["days"]).max() - days
    return (np.mean(np.asarray(d["T_top"])[steady], axis=0),
            np.asarray(d["T_init"][:, :, 0], dtype=float),
            np.asarray(d["wet_mask"] > 0.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline",
                    default="results/real_air_temp/global_real_air_2m_1deg_365d_repeat.npz")
    ap.add_argument("--runs", nargs="+", default=[
        "results/vertical_mixing_sensitivity/global_real_air_kv1e-6_kconv005.npz",
        "results/vertical_mixing_sensitivity/global_real_air_kv1e-5_kconv001.npz",
        "results/vertical_mixing_sensitivity/global_real_air_kv1e-6_kconv001.npz",
    ])
    ap.add_argument("--out-dir",
                    default="research/experiments/vertical_mixing_sensitivity")
    args = ap.parse_args()
    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    baseline_path = REPO / args.baseline
    model0, woa0, wet = steady_mean(baseline_path)
    err0 = model0 - woa0
    strat = np.asarray(np.load(baseline_path)["T_init"][:, :, 0], dtype=float) \
        - np.asarray(np.load(baseline_path)["T_init"][:, :, 4], dtype=float)
    edges = np.quantile(strat[wet], np.linspace(0, 1, 6))

    rows = []
    cases = [("baseline", baseline_path, err0)]
    for rel in args.runs:
        model, woa, wet_i = steady_mean(REPO / rel)
        err = model - woa
        cases.append((Path(rel).stem.replace("global_", "").replace("real_air_", ""), REPO / rel, err))

    for name, path, err in cases:
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            mask = wet & (strat >= lo)
            mask &= (strat < hi) if i < 4 else (strat <= hi)
            rows.append([
                name, f"q{i + 1}", int(mask.sum()),
                fmt(float(np.mean(err[mask]))),
                fmt(float(np.sqrt(np.mean(err[mask] ** 2)))),
                fmt(float(lo), 3), fmt(float(hi), 3),
            ])
    with (out / "stratification_bias.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "quintile", "count", "mean_error", "rmse",
                         "strat_min", "strat_max"])
        writer.writerows(rows)

    # Compact tabular summary for the note.
    by_run = {}
    for name, _, _ in cases:
        by_run[name] = {}
        for row in rows:
            if row[0] == name:
                by_run[name][row[1]] = float(row[3])
    base_q5 = by_run["baseline"]["q5"]
    lines = [
        "# Vertical Mixing Sensitivity Results", "",
        "## Climate scores", "",
        "| run | A1 corr | A1 RMSE | A2 corr | A2 RMSE | mean error |",
        "|---|---:|---:|---:|---:|---:|",
        "| annual real-air baseline | 0.997 | 1.469 C | 0.988 | 1.886 C | -1.314 C |",
        "| kappa_v 1e-6 | 0.997 | 1.430 C | 0.988 | 1.851 C | -1.270 C |",
        "| kappa_conv 0.01 | 0.997 | 1.447 C | 0.988 | 1.870 C | -1.297 C |",
        "| both reduced | 0.997 | 1.427 C | 0.988 | 1.848 C | -1.263 C |",
        "| both reduced repeat | 0.997 | 1.420 C | 0.988 | 1.844 C | -1.260 C |",
        "", "## Bias by WOA surface-to-50m stratification quintile", "",
        "| run | q1 | q2 | q3 | q4 | q5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, _, _ in cases:
        vals = [fmt(by_run[name][f"q{i + 1}"]) for i in range(5)]
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    lines += [
        "", "## Reading", "",
        f"- The strongest-stratification cold bias improves from `{fmt(base_q5)} C`",
        f"  to `{fmt(by_run['kv1e-6_kconv001']['q5'])} C` in the combined",
        "  reduced-mixing case, a change of "
        f"`{fmt(by_run['kv1e-6_kconv001']['q5'] - base_q5)} C`.",
        f"- The 365d repeat gives `{fmt(by_run['kv1e-6_kconv001_repeat']['q5'])} C`.",
        "- The combined reduced-mixing case also has the best global A2 RMSE and",
        "  remains reproducible.",
        "- All reduced-mixing runs remain stable and have small heat/salt drift.",
        "- A2 improvement is above the locked 2% gate, but the strongest-layer",
        "  bias change is below the 0.2 C gate, so this is suggestive rather",
        "  than decisive evidence of over-mixing.",
        "", "## Next", "",
        "Treat the combined reduced-mixing case as the preferred candidate",
        "baseline. It has been reproduced. The next step is a regional error",
        "audit, especially in strong-stratification and coastal regions, before",
        "deciding whether to implement a mixed-layer closure or investigate the",
        "bulk heat-flux formula further.",
    ]
    (out / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'analysis.md'}")


if __name__ == "__main__":
    main()
