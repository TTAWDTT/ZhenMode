"""Prescribed verified real-weather stress over the same controlled water domain."""

from pathlib import Path

import numpy as np

from zhenmode.benchmarks.channel_dynamics import contract as channel_contract
from zhenmode.benchmarks.standing_wave import digest
from zhenmode.preparation.wind_sample import validate_sample
from zhenmode.provenance.sources import load_json, sha256_file


def contract(stress_file, method="baseline"):
    path = Path(stress_file).resolve()
    sample = load_json(path)
    validate_sample(sample)
    if (
        sample.get("data_kind")
        not in ("observed_weather_derived_stress", "manufactured_weather_derived_stress")
        or sample.get("schema") != "jra-point-mean-stress-v2"
    ):
        raise ValueError("requires identified weather derivation")
    if sample["source_dates"] != [
        "1958-01-01 00:00:00",
        "1958-01-01 03:00:00",
        "1958-01-01 06:00:00",
    ]:
        raise ValueError("requires the frozen source six-hour window")
    tx, ty = sample["tau_x_N_m2"], sample["tau_y_N_m2"]
    if any(type(v) not in (int, float) or not np.isfinite(v) or abs(v) > 4 for v in (tx, ty)):
        raise ValueError("invalid SI mean stress")
    c = channel_contract("geostrophic-adjustment", "coarse", method)
    c.update(
        schema="real-wind-channel-v1",
        experiment="real-wind-control"
        if sample["data_kind"].startswith("observed")
        else "manufactured-wind-control",
        data_kind=sample["data_kind"],
        duration_s=21600.0,
        output_s=900.0,
        f=float(2 * 7.2921e-5 * np.sin(np.deg2rad(sample["actual_point"]["latitude"]))),
        wind_stress=dict(tau_x_N_m2=float(tx), tau_y_N_m2=float(ty)),
        stress_source=dict(path=str(path), sha256=sha256_file(path)),
        reference=dict(
            kind="state_and_budget_comparison",
            accuracy_reference="none; no trajectory error ranking",
            scope="real-weather derived constant mean stress; idealized flat warm channel",
        ),
    )
    c["explicitly_zero"].remove("wind")
    return c


def validate_contract(c):
    if not isinstance(c, dict) or c.get("schema") != "real-wind-channel-v1":
        raise ValueError("unknown forced channel")
    source = c["stress_source"]
    if sha256_file(source["path"]) != source["sha256"]:
        raise ValueError("derived real stress changed")
    expected = contract(source["path"], c.get("method", "baseline"))
    if digest(c) != digest(expected):
        raise ValueError("forced channel differs from frozen source/configuration")
