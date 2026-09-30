"""Bounded cross-process replay against the frozen actual material trajectory."""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--stop-day", type=int, choices=(15, 30), required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "src"))
    import jax
    import jax.numpy as jnp
    import numpy as np

    from grid import GlobalOceanGrid
    from jax_solver_global import FDPhysParams, JaxStateG
    from material_top import make_material_top_restart_contract, make_material_top_step
    from restart_contract import file_sha256, load_restart, save_restart

    if jax.default_backend() != "gpu":
        raise ValueError("registered replay requires the original CUDA execution contract")
    protocol_path = Path(__file__).with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    baseline = root / protocol["baseline"]
    manifest_path = baseline / "manifest_start.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_protocol_path = baseline.parent / "protocol.json"
    original_protocol = json.loads(original_protocol_path.read_text(encoding="utf-8"))
    frozen = {**manifest["source_sha256"], **manifest["input_sha256"]}
    for name, digest in frozen.items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"original source or input changed: {name}")
    scalar = manifest["scalar_params"]
    with np.load(baseline / "parameter_arrays_used.npz", allow_pickle=False) as arrays:
        params = FDPhysParams(**{name: jnp.asarray(arrays[name]) if name in arrays.files else scalar[name]
                                 for name in FDPhysParams._fields})
    with np.load(baseline / "grid_used.npz", allow_pickle=False) as arrays:
        grid = GlobalOceanGrid(**{name: arrays[name].item() if arrays[name].ndim == 0 else arrays[name].copy()
                                  for name in arrays.files})
    contract = make_material_top_restart_contract(
        grid, params, forcing={"raw_input_sha256": manifest["input_sha256"]},
        controls={"calendar": "fixed_Jan2023", "stops": original_protocol["stops"]},
        execution={"backend": jax.default_backend(), "device": str(jax.devices()[0]), "jax": jax.__version__},
        subcycle_scheme="actual_geometry_v2", max_subcycles=128,
        momentum_diffusion_scheme="joint_heun_v1")
    start_path = arguments.start.resolve()
    original = load_restart(start_path, contract)
    final_step = int(arguments.stop_day * 86400. / params.dt)
    required_start_day = 7 if arguments.stop_day == 15 else 15
    if (original.step != int(required_start_day * 86400. / params.dt)
            or original.counters != {"accepted_steps": original.step}
            or final_step * params.dt != arguments.stop_day * 86400.):
        raise ValueError("checkpoint does not match the registered segment")
    target = arguments.output.resolve()
    target.mkdir(parents=True, exist_ok=False)
    replay_sources = [Path(__file__).resolve(), protocol_path, manifest_path, original_protocol_path,
                      baseline / "parameter_arrays_used.npz", baseline / "grid_used.npz", start_path]
    audit_hashes = {str(path): file_sha256(path) for path in replay_sources}
    (target / "manifest.json").write_text(json.dumps({"original_sources_inputs": frozen,
        "replay_inputs": audit_hashes, "contract": contract}, indent=2) + "\n", encoding="utf-8")
    for path in [Path(__file__).resolve(), protocol_path, *(root / "src" / f"{name}.py" for name in contract["sources"])]:
        destination = target / "sources" / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    state = JaxStateG(**{name: jnp.asarray(value) for name, value in original.state.items()})
    totals = {name: value.copy() for name, value in original.cumulative.items()}
    history = original.history["minimum_thickness_m"].tolist()
    if len(history) != original.step or set(original.history) != {"minimum_thickness_m"}:
        raise ValueError("checkpoint history does not match the registered trajectory")
    step = make_material_top_step(params, subcycle_scheme="actual_geometry_v2", max_subcycles=128,
                                  momentum_diffusion_scheme="joint_heun_v1")
    started = time.perf_counter()
    compiled = step.lower(state).compile()
    report = {"status": "running", "start_step": original.step, "stop_step": final_step,
              "compile_seconds_not_benchmark": time.perf_counter() - started, "records": [],
              "scope": protocol["scope"], "start_checkpoint_sha256": file_sha256(start_path)}

    def save_report():
        (target / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save_report()
    for accepted in range(original.step + 1, final_step + 1):
        result = compiled(state)
        jax.block_until_ready(result)
        checks = {name: bool(value) if np.asarray(value).dtype.kind == "b" else float(value)
                  for name, value in result.checks.items()}
        if not bool(result.valid):
            report.update(status="first_rejection", first_rejected_absolute_attempt=accepted,
                          checks={name: value if isinstance(value, bool) or np.isfinite(value) else None
                                  for name, value in checks.items()})
            for name in state._fields:
                if np.asarray(getattr(result.state, name)).tobytes() != np.asarray(getattr(state, name)).tobytes():
                    raise AssertionError("rejected step did not roll back all fields")
            np.savez_compressed(target / "rejected_state.npz", **{name: np.asarray(value) for name, value in result.attempted_state._asdict().items()})
            np.savez_compressed(target / "rejected_budget.npz", **{name: np.asarray(value) for name, value in result.budget.items()})
            save_report()
            raise SystemExit(1)
        state = result.state
        for name, value in result.budget.items():
            totals[name] += np.asarray(value)
        history.append(checks["minimum_wet_thickness_m"])
        if accepted % 144 == 0:
            record = {"accepted_steps": accepted, "actual_days": accepted * params.dt / 86400., "checks": checks}
            report["records"].append(record)
            save_report()
            print(json.dumps(record, allow_nan=False), flush=True)
    checkpoint = target / "strict_restart.npz"
    save_restart(checkpoint, state, contract, step=final_step, counters={"accepted_steps": final_step},
                 cumulative=totals, history={"minimum_thickness_m": history})
    restored = load_restart(checkpoint, contract)
    report["strict_roundtrip_fields_equal"] = {
        name: np.asarray(getattr(state, name)).tobytes() == restored.state[name].tobytes() for name in state._fields}
    report["strict_roundtrip_budgets_equal"] = {
        name: value.tobytes() == restored.cumulative[name].tobytes() for name, value in totals.items()}
    report["strict_roundtrip_history_equal"] = np.asarray(history).tobytes() == restored.history["minimum_thickness_m"].tobytes()
    report["strict_roundtrip_counters_equal"] = restored.counters == {"accepted_steps": final_step}
    checks_pass = (all(report["strict_roundtrip_fields_equal"].values())
                   and all(report["strict_roundtrip_budgets_equal"].values())
                   and report["strict_roundtrip_history_equal"] and report["strict_roundtrip_counters_equal"])
    if arguments.stop_day == 30:
        reference = load_restart(baseline / "strict_restart_target30d.npz", contract)
        report["continuous_30day_fields_byte_equal"] = {
            name: restored.state[name].tobytes() == reference.state[name].tobytes() for name in state._fields}
        report["continuous_30day_budgets_byte_equal"] = {
            name: restored.cumulative[name].tobytes() == reference.cumulative[name].tobytes() for name in totals}
        report["continuous_30day_history_byte_equal"] = {
            name: restored.history[name].tobytes() == reference.history[name].tobytes() for name in restored.history}
        report["continuous_30day_clock_counters_equal"] = (
            restored.step == reference.step and restored.elapsed_seconds == reference.elapsed_seconds
            and restored.counters == reference.counters and restored.outputs == reference.outputs)
        checks_pass = (checks_pass and all(report["continuous_30day_fields_byte_equal"].values())
                       and all(report["continuous_30day_budgets_byte_equal"].values())
                       and all(report["continuous_30day_history_byte_equal"].values())
                       and report["continuous_30day_clock_counters_equal"])
    report["source_input_hashes_unchanged"] = all(file_sha256(root / name) == digest for name, digest in frozen.items())
    report["replay_input_hashes_unchanged"] = all(file_sha256(name) == digest for name, digest in audit_hashes.items())
    checks_pass = checks_pass and report["source_input_hashes_unchanged"] and report["replay_input_hashes_unchanged"]
    report["status"] = "registered_segment_passed_not_industrial_qualification" if checks_pass else "replay_mismatch"
    report["elapsed_seconds_not_benchmark"] = time.perf_counter() - started
    save_report()
    print(json.dumps({name: value for name, value in report.items() if name != "records"}, indent=2), flush=True)
    if not checks_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
