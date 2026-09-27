"""Validate standardized benchmark run contracts.

The benchmark protocol requires every run to state whether the grid, bathymetry,
initial state, forcing, sea-ice treatment, duration, scoring, and provenance are
comparable.  Missing or unspecified fields are treated as not_comparable
rather than being silently accepted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_SECTIONS: tuple[str, ...] = (
    "model",
    "run_id",
    "status",
    "grid",
    "bathymetry",
    "initial_state",
    "forcing",
    "sea_ice",
    "duration",
    "scoring",
    "provenance",
)

STATUS_SECTIONS: tuple[str, ...] = (
    "grid", "bathymetry", "initial_state", "forcing", "sea_ice", "duration",
    "scoring", "provenance")

ALLOWED_STATUS = {"comparable", "not_comparable", "pre_registered"}


def _status(section: Any) -> str | None:
    if isinstance(section, dict):
        value = section.get("status")
        if isinstance(value, str):
            return value
    return None


def validate_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return a closed-world contract audit for one benchmark manifest.

    contract_pass means the manifest explicitly states every required
    section and every section has a declared comparability status.  A section
    may still be not_comparable; that keeps the gap honest instead of
    dropping it from the comparison.
    """
    checks: dict[str, bool] = {}
    missing: list[str] = []
    not_comparable: list[str] = []

    for key in REQUIRED_SECTIONS:
        checks[f"{key}_present"] = key in manifest
        if key not in manifest:
            missing.append(key)
            if key in STATUS_SECTIONS:
                not_comparable.append(f"{key}:missing_status")
            continue

        if key not in STATUS_SECTIONS:
            checks[f"{key}_status_declared"] = True
            continue

        status = _status(manifest.get(key))
        if status is None:
            checks[f"{key}_status_declared"] = False
            not_comparable.append(f"{key}:missing_status")
        elif status not in ALLOWED_STATUS:
            checks[f"{key}_status_declared"] = False
            not_comparable.append(f"{key}:invalid_status")
        else:
            checks[f"{key}_status_declared"] = True
            if status == "not_comparable":
                not_comparable.append(key)

    return {
        "contract_pass": all(checks.values()),
        "checks": checks,
        "missing_sections": missing,
        "not_comparable_sections": not_comparable,
        "required_sections": list(REQUIRED_SECTIONS),
    }


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="+")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    reports = [validate_contract(json.loads(Path(p).read_text(encoding="utf-8")))
               for p in args.manifest]
    result = {"contract_pass": all(x["contract_pass"] for x in reports),
              "reports": reports}
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
