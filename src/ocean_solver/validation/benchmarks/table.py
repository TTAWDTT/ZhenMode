"""Render a compact comparison table from benchmark JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _label(path: str | Path, explicit: str | None) -> str:
    return explicit or Path(path).parent.name + "/" + Path(path).stem


def load_metric(path: str | Path, label: str | None = None) -> dict:
    """Read one benchmark JSON and flatten the fields used in reports."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    global_metrics = data.get("global") or data
    return {
        "label": _label(path, label),
        "path": str(path),
        "verdict": data.get("verdict", "MISSING"),
        "metric_definition": data.get("metric_definition", "legacy_equal_cell_index_box_v1"),
        "days_end": data.get("days_end", ""),
        "global_a2_rmse": data.get("global_a2_rmse_c"),
        "na_raw_rmse": data.get("north_atlantic_40_60", {}).get("raw_rmse"),
        "near_wall_raw_rmse": data.get("near_wall_55_60", {}).get("raw_rmse"),
        "raw_bias": global_metrics.get("raw_bias"),
        "raw_rmse": global_metrics.get("raw_rmse"),
        "heat_drift": data.get("heat_drift_percent"),
        "salt_drift": data.get("salt_drift_percent"),
        "frozen_cells": data.get("ice", {}).get("n_cells"),
    }


def markdown_table(rows: list[dict]) -> str:
    """Render rows as a compact benchmark comparison table."""
    headers = ["run", "metric definition", "verdict", "days", "global A2", "NA RMSE",
               "near-wall RMSE", "global bias", "heat drift", "salt drift"]
    fields = ["label", "metric_definition", "verdict", "days_end", "global_a2_rmse", "na_raw_rmse",
              "near_wall_raw_rmse", "raw_bias", "heat_drift", "salt_drift"]
    lines = ["| " + " | ".join(headers) + " |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        cells = []
        for field in fields:
            value = row.get(field)
            if value is None:
                value = ""
            elif isinstance(value, float):
                value = f"{value:.4g}"
            cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare standardized benchmark JSON files.")
    parser.add_argument("json", nargs="+", metavar="JSON")
    parser.add_argument("--label", action="append", default=[],
                        metavar="LABEL=PATH",
                        help="display label for a JSON path")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    labels: dict[str, str] = {}
    for item in args.label:
        key, value = item.split("=", 1)
        labels[str(Path(value))] = key
    rows = [load_metric(path, labels.get(str(Path(path))))
            for path in args.json]
    table = markdown_table(rows)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
    print(table)




if __name__ == "__main__":
    main()
