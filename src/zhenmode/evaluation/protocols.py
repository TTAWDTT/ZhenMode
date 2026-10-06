"""Strict, versioned contracts; no metric computation lives here."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from zhenmode.provenance.sources import load_json as load_json

V2 = "area_weighted_angular_box_v2"
LEGACY = "legacy_equal_cell_index_box_v1"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()




def _keys(value: dict, fields: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields must be exactly {sorted(fields)}")


def validate_protocol(value: dict) -> dict:
    _keys(value, {"schema_version", "id", "kind", "case_id", "metric_definition",
                  "reference", "window", "spatial", "temporal", "temperature_unit",
                  "acceptance", "limitations"}, "protocol")
    if value["schema_version"] != 1 or value["kind"] != "sst":
        raise ValueError("supported evaluation protocol is schema_version=1, kind=sst")
    for name in ("id", "case_id"):
        if not isinstance(value[name], str) or not value[name]:
            raise ValueError(f"{name} must be a stable nonempty identity")
    if value["metric_definition"] != V2:
        raise ValueError("new scoring requires area v2; historical v1 is imported unchanged")
    if value["temperature_unit"] != "degC":
        raise ValueError("SST scoring requires degC; no implicit Kelvin conversion")
    _keys(value["window"], {"start_day", "end_day"}, "window")
    start, end = value["window"]["start_day"], value["window"]["end_day"]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
           for x in (start, end)) or not 0 <= start < end:
        raise ValueError("window must be finite days with 0 <= start_day < end_day")
    if value["spatial"] != {"coordinate_system": "geographic_degrees",
                            "mask": "reference_wet", "weighting": "wet_cell_area",
                            "remapping": "none"}:
        raise ValueError("shared geographic grid, wet area weighting and no remapping required")
    if value["temporal"] != {"weighting": "arithmetic_saved_records"}:
        raise ValueError("only existing arithmetic saved-record temporal scoring is implemented")
    _keys(value["reference"], {"role", "version", "sha256"}, "reference")
    if value["reference"]["role"] != "saved_initial_surface_temperature":
        raise ValueError("existing scorer references saved initialization, not independent observations")
    if not isinstance(value["reference"]["version"], str) or not value["reference"]["version"]:
        raise ValueError("reference version must be explicit")
    sha = value["reference"]["sha256"]
    if sha is not None and (not isinstance(sha, str) or len(sha) != 64
                            or any(c not in "0123456789abcdef" for c in sha)):
        raise ValueError("reference sha256 must be a 64-digit lowercase hash or null")
    _keys(value["acceptance"], {"max_raw_rmse_degC", "max_abs_heat_drift_percent",
                              "max_abs_salt_drift_percent"}, "acceptance")
    for name, limit in value["acceptance"].items():
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, (int, float))
                                  or not math.isfinite(limit) or limit < 0):
            raise ValueError(f"{name} must be null or a finite nonnegative declared limit")
    if not isinstance(value["limitations"], list) or not all(
            isinstance(x, str) for x in value["limitations"]):
        raise ValueError("limitations must be a list of strings")
    return value


def load_protocol(path: str | Path) -> dict:
    return validate_protocol(load_json(path))
