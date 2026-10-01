# ruff: noqa: E402
"""Serial registered four-cell short window; stop globally at first rejection."""

import os

os.environ.update(
    JAX_PLATFORMS="cpu",
    JAX_ENABLE_X64="true",
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
)
import argparse
import hashlib
import json
import time
from io import BytesIO
from pathlib import Path

import numpy as np

folder = Path(__file__).parent
parent = None
parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--protocol-sha256", required=True)
parser.add_argument("--protocol", type=Path, required=True)
parser.add_argument("--source-dir", type=Path, required=True)
parser.add_argument("--archive-dir", type=Path, required=True)
parser.add_argument("--evidence-dir", type=Path, required=True)
parser.add_argument("--donor-provenance", type=Path, required=True)
parser.add_argument("--raw-t", type=Path, required=True)
parser.add_argument("--raw-s", type=Path, required=True)
args = parser.parse_args()
run = args.run_dir
parent = args.evidence_dir


def sha(content):
    return hashlib.sha256(content).hexdigest()


protocol_bytes = args.protocol.read_bytes()
assert sha(protocol_bytes) == args.protocol_sha256
protocol = json.loads(protocol_bytes)


def safe(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe(item) for item in value]
    return value


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(safe(value), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


cases = protocol["cells_order"]
spent = {case: 0.0 for case in cases}


def progress(case=None, started=None, completed=0):
    save_json(
        run / "progress.json",
        {
            "active_case": case,
            "active_start_perf_counter": started,
            "spent_case_seconds": spent,
            "completed_common_round": completed,
        },
    )


progress()
startup = time.perf_counter()
for name, digest in protocol["historical_source_hashes"].items():
    expected = protocol["instrumented_material_sha256"] if name == "src/material_top.py" else digest
    assert sha((args.source_dir / Path(name).name).read_bytes()) == expected, name
assert (
    sha((parent / "full_crop_contract.json").read_bytes()) == protocol["full_crop_contract_sha256"]
)
archive = args.archive_dir


def load(name):
    content = (archive / name).read_bytes()
    assert sha(content) == protocol["base_inputs"][name], name
    if name.endswith(".json"):
        return json.loads(content)
    with np.load(BytesIO(content), allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


arrays = load("parameter_arrays_used.npz")
grid = load("grid_used.npz")
original = load("initial_state_used.npz")
manifest = load("manifest_start.json")
for label, identity in protocol["raw_identities"].items():
    with (args.raw_t if label == "T" else args.raw_s).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == identity["sha256"]
donor_path = args.donor_provenance
donor_bytes = donor_path.read_bytes()
assert sha(donor_bytes) == protocol["donor_provenance_sha256"]
donors = json.loads(donor_bytes)["connected_depth_donors"]
assert len(donors) == 440
perturbed = {key: value.copy() for key, value in original.items()}
for donor in donors:
    index = tuple(donor["index"])
    for label in ("T", "S"):
        perturbed[label][index] += donor[label + "_change"]
import sys

sys.path.insert(0, str(args.source_dir))
import jax
import jax.numpy as jnp

import jax_solver_global as solver
from material_top import make_observed_material_top_step, material_inventory

assert jax.default_backend() == "cpu" and jax.config.jax_enable_x64
params0 = solver.FDPhysParams(
    **{
        key: jnp.asarray(arrays[key]) if key in arrays else manifest["scalar_params"][key]
        for key in solver.FDPhysParams._fields
    }
)
cropped = {
    key: (
        value[:, 1:].copy()
        if value.ndim >= 2 and value.shape[:2] == (360, 132)
        else value[1:].copy()
        if value.shape == (132,)
        else value.copy()
    )
    for key, value in arrays.items()
}
cropped["interior_mask_z"][:, 0] = 0.0
params1 = params0._replace(**{key: jnp.asarray(value) for key, value in cropped.items()}, ny=131)
diagonal = solver._column_projection_diagonal(params1)
rebuilt = np.asarray(1.0 / jnp.where(diagonal > 0, diagonal, 1.0))
kept = cropped["projection_inv_diagonal"].copy()
kept[:, 0] = rebuilt[:, 0]
assert kept[:, 1:].tobytes() == cropped["projection_inv_diagonal"][:, 1:].tobytes()
params1 = params1._replace(projection_inv_diagonal=jnp.asarray(kept))
parameters = {0: params0, 1: params1}
functions = {
    boundary: make_observed_material_top_step(params) for boundary, params in parameters.items()
}
compiled = {}
states = {}
for case in cases:
    data = original if case[1] == "0" else perturbed
    states[case] = solver.JaxStateG(
        **{
            key: jnp.asarray(value[:, 1:] if case[3] == "1" else value)
            for key, value in data.items()
        }
    )
area = grid["dx_2d"] * float(grid["dy"])
wet = arrays["wet_mask"] > 0
mask_bytes = (parent / "four_cell_private_deltas.npz").read_bytes()
assert sha(mask_bytes) == protocol["private_prerequisites"]["four_cell_private_deltas.npz"]
with np.load(BytesIO(mask_bytes), allow_pickle=False) as masks:
    retained = masks["retained_low_eta_mask"].copy()
    original609 = masks["low_eta_mask"].copy()
south = np.zeros_like(wet)
south[:, 2:6] = wet[:, 2:6]
north = np.zeros_like(wet)
north[:, 8:16] = wet[:, 8:16]
new_wall = np.zeros_like(wet)
new_wall[:, 1] = wet[:, 1]
new_constrained161 = np.zeros_like(wet)
new_constrained161[:, 1] = original609[:, 1]
deleted152 = np.zeros_like(wet)
deleted152[:, 0] = original609[:, 0]


def checkpoint(path, state):
    with path.open("xb") as stream:
        np.savez_compressed(
            stream, **{key: np.asarray(value) for key, value in state._asdict().items()}
        )


initial_dir = run / "round_000"
initial_dir.mkdir()
for case in cases:
    checkpoint(initial_dir / (case + ".npz"), states[case])
save_json(initial_dir / "COMMITTED.json", {"round": 0, "model_seconds": 0, "cases": cases})
spent = {case: (time.perf_counter() - startup) / 4.0 for case in cases}
records = run / "attempt_records.jsonl"
trajectory = run / "trajectory.jsonl"


def append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(safe(value), allow_nan=False) + "\n")
        stream.flush()


def region(eta, mask, crop):
    selected = mask[:, crop:]
    weights = area[:, crop:][selected]
    values = eta[selected]
    return {
        "count": int(selected.sum()),
        "area_m2": float(weights.sum()),
        "mean_m": float(np.sum(weights * values) / weights.sum()),
        "rms_m": float(np.sqrt(np.sum(weights * values**2) / weights.sum())),
        "minimum_m": float(values.min()),
        "maximum_m": float(values.max()),
    }


accepted_rounds = 0
reason = None
for round_id in range(1, 13):
    pending = {}
    metrics = {}
    staged_dir = run / ("round_" + str(round_id).zfill(3))
    staged_dir.mkdir()
    for case in cases:
        started = time.perf_counter()
        progress(case, started, accepted_rounds)
        boundary = int(case[3])
        lower_seconds = compile_seconds = 0.0
        cache_reused = boundary in compiled
        if not cache_reused:
            begin = time.perf_counter()
            lowered = functions[boundary].lower(states[case])
            lower_seconds = time.perf_counter() - begin
            begin = time.perf_counter()
            compiled[boundary] = lowered.compile()
            compile_seconds = time.perf_counter() - begin
        before = states[case]
        begin = time.perf_counter()
        result, observed = compiled[boundary](before)
        jax.block_until_ready((result, observed))
        execution_seconds = time.perf_counter() - begin
        checks = {key: np.asarray(value).item() for key, value in result.checks.items()}
        budget = {key: np.asarray(value).tolist() for key, value in result.budget.items()}
        diagnostic = {key: np.asarray(value).item() for key, value in observed.items()}
        attempted = result.attempted_state
        returned = result.state
        eta = np.asarray(attempted.eta)
        incoming_inventory = np.asarray(material_inventory(before, parameters[boundary])).tolist()
        attempted_inventory = np.asarray(
            material_inventory(attempted, parameters[boundary])
        ).tolist()
        match = {"required": round_id == 1, "passed": True}
        if round_id == 1:
            reference_file = parent / (
                "full_baseline_attempted_private.npz"
                if case == "I0B0"
                else case + "-full_attempted_private.npz"
            )
            with np.load(reference_file, allow_pickle=False) as reference:
                differences = {
                    key: float(np.max(abs(np.asarray(value) - reference[key])))
                    for key, value in attempted._asdict().items()
                }
                ice_equal = np.asarray(attempted.ice).tobytes() == reference["ice"].tobytes()
            reference_summary = json.loads(
                (
                    parent
                    / (
                        "full_baseline_summary.json"
                        if case == "I0B0"
                        else case + "-full_summary.json"
                    )
                ).read_text()
            )
            flags_equal = bool(result.valid) == reference_summary["valid"] and all(
                checks[key] == value
                for key, value in reference_summary["checks"].items()
                if isinstance(value, bool)
            )
            match.update(
                max_absolute_field_differences=differences,
                ice_byte_equal=ice_equal,
                flags_equal=flags_equal,
                passed=ice_equal
                and flags_equal
                and all(value <= 1e-11 for value in differences.values()),
            )
        valid = bool(result.valid)
        metric = {
            "case": case,
            "round": round_id,
            "model_seconds": round_id * 300,
            "valid": valid,
            "node_eta_m": float(eta[302, 2 - boundary]),
            "retained296": region(eta, retained, boundary),
            "south_j2_to5": region(eta, south, boundary),
            "north_j8_to15": region(eta, north, boundary),
            "new_wall_geographic_j1": region(eta, new_wall, boundary),
            "new_constrained161": region(eta, new_constrained161, boundary),
            "deleted152": region(eta, deleted152, 0)
            if boundary == 0
            else {"absent_from_domain": True, "count": 152, "not_credited_as_improvement": True},
            "checks": checks,
            "budget": budget,
            "observed": diagnostic,
            "incoming_inventory": incoming_inventory,
            "attempted_inventory": attempted_inventory,
            "returned_equals_attempt": {
                key: np.asarray(value).tobytes() == np.asarray(getattr(attempted, key)).tobytes()
                for key, value in returned._asdict().items()
            },
            "returned_equals_incoming": {
                key: np.asarray(value).tobytes() == np.asarray(getattr(before, key)).tobytes()
                for key, value in returned._asdict().items()
            },
            "nonfinite_attempt_counts": {
                key: int(np.count_nonzero(~np.isfinite(np.asarray(value))))
                for key, value in attempted._asdict().items()
            },
            "instrumentation_match": match,
            "lower_seconds": lower_seconds,
            "compile_seconds": compile_seconds,
            "execution_seconds": execution_seconds,
            "kernel_reused": cache_reused,
            "actual_internal_face_identity": "oldj0/j1"
            if boundary == 0
            else "oldj1/j2; closed southern exterior iszero",
            "phase_diagnostics_limit": "Scalar checkpoint observations, not force ablation; earlier1.83e7 value remains historical inference.",
        }
        checkpoint(staged_dir / (case + "-attempted.npz"), attempted)
        if not valid:
            checkpoint(staged_dir / (case + "-returned.npz"), returned)
        spent[case] += time.perf_counter() - started
        metric["case_cumulative_wall_seconds"] = spent[case]
        metric["round_status"] = "staged"
        append(records, metric)
        if not valid:
            reason = {
                "type": "original_gate_rejection",
                "case": case,
                "round": round_id,
                "all_six_rollback_byte_equal": all(metric["returned_equals_incoming"].values()),
            }
            break
        if not match["passed"]:
            reason = {
                "type": "instrumentation_numerical_match_failed",
                "case": case,
                "round": round_id,
            }
            break
        if spent[case] >= 900:
            reason = {"type": "case_budget_exceeded", "case": case, "round": round_id}
            break
        pending[case] = returned
        metrics[case] = metric
    if reason:
        break
    states = pending
    accepted_rounds = round_id
    save_json(
        staged_dir / "COMMITTED.json",
        {
            "round": round_id,
            "model_seconds": round_id * 300,
            "cases": cases,
            "cells": metrics,
            "checkpoint_sha256": {
                case: sha((staged_dir / (case + "-attempted.npz")).read_bytes()) for case in cases
            },
        },
    )
    append(
        trajectory,
        {
            "round": round_id,
            "model_seconds": round_id * 300,
            "cells": metrics,
            "status": "committed_common_round",
        },
    )
    progress(completed=accepted_rounds)
    print("COMMITTED_COMMON_ROUND", round_id, flush=True)
save_json(
    run / "summary.json",
    {
        "completed_common_rounds": accepted_rounds,
        "model_seconds": accepted_rounds * 300,
        "stop_reason": reason,
        "all12_completed": accepted_rounds == 12,
        "case_cumulative_wall_seconds": spent,
        "persistent_cache_used": False,
        "shape_kernel_count": len(compiled),
        "partial_round_excluded_from_four_cell_effects": reason is not None,
    },
)
progress(completed=accepted_rounds)
print("SHORT_WINDOW_END", accepted_rounds, reason, flush=True)
