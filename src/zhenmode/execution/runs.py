"""Create-only run directories and explicit execution/provenance records."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from zhenmode.provenance.sources import sha256_file as file_hash

from .resolve import canonical_hash, resource_estimate
from .resources import run_cuda_worker, validate_budget
from .schema import ConfigurationError, reference_path


def write_json(path, value, *, create=False):
    with Path(path).open("x" if create else "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")


def source_identity():
    import zhenmode

    package = Path(zhenmode.__file__).resolve().parent
    return {path.relative_to(package).as_posix(): file_hash(path) for path in sorted(package.rglob("*.py"))}


def environment_identity():
    packages = {}
    for name in ("zhenmode", "jax", "jaxlib", "jax-cuda13-plugin", "jax-cuda13-pjrt", "numpy", "scipy", "netCDF4", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(), "packages": packages,
            "flags": {name: os.environ.get(name) for name in ("JAX_PLATFORMS", "XLA_FLAGS", "JAX_ENABLE_X64", "OCEAN_PAV_NITER", "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION")}}


def git_identity(root):
    def read(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": read("rev-parse", "HEAD"), "branch": read("branch", "--show-current"),
            "working_tree_status": read("status", "--porcelain")}


def data_identity(case, root, *, require_files):
    result = {}
    for data in case["data"]:
        path = reference_path(root, data["path"])
        exists = path.is_file()
        if require_files and not exists:
            raise ConfigurationError(f"missing case input {data['role']}: {path}; source: {data['source']}")
        observed = file_hash(path) if exists else None
        mismatch = bool(observed and data["sha256"] and observed != data["sha256"])
        if mismatch and require_files:
            raise ConfigurationError(f"case input checksum mismatch: {data['role']} ({path})")
        result[data["role"]] = {**data, "resolved_path": str(path), "observed_sha256": observed,
                                "verification": "checksum_mismatch" if mismatch else "verified" if observed and data["sha256"] else "observed_unpinned" if observed else "pending"}
    if case["grid"]["kind"] == "synthetic":
        result["generated"] = {"version": "synthetic-case-services-v1", "observed_sha256": canonical_hash(case), "verification": "generated_from_frozen_case"}
    return result


def physical_problem_identity(case):
    # Includes shared grid: current scoring expressly has no hidden regridding.
    return canonical_hash({name: case[name] for name in ("domain", "grid", "initial", "forcing", "duration", "output_interval")})


def create_run(expanded, root, outputs):
    outputs = Path(outputs).resolve()
    outputs.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{timestamp}-{expanded['config_hash'][:12]}-{uuid.uuid4().hex[:8]}"
    directory = outputs / run_id
    directory.mkdir(exist_ok=False)
    data = data_identity(expanded["case"], root, require_files=False)
    manifest = {
        "schema_version": 1, "run_id": run_id, "experiment_id": expanded["experiment_id"],
        "name_zh": expanded["name_zh"], "purpose": expanded["purpose"],
        "case_id": expanded["case_id"], "method": expanded["method"], "preset_id": expanded["preset_id"],
        "variant": expanded["variant"], "changes": expanded["changes"], "config_hash": expanded["config_hash"],
        "evaluation_protocol": expanded["evaluation_protocol"], "protocol_sha256": expanded["protocol_sha256"],
        "protocol_file_sha256": expanded["protocol_file_sha256"], "protocol_content_sha256": expanded["protocol_content_sha256"],
        "execution_status": "proposed", "comparability": {"status": "not_assessed", "limitations": expanded["case"]["comparison"]["limitations"]},
        "acceptance": {"status": "not_assessed", "reason": "execution success does not imply metric acceptance"},
        "created_at": datetime.now(timezone.utc).isoformat(), "source_identity": source_identity(),
        "git": git_identity(root), "data": data,
        "data_sha256": canonical_hash({key: value["observed_sha256"] for key, value in data.items()}),
        "physical_problem_sha256": physical_problem_identity(expanded["case"]),
        "effective_physics_sha256": canonical_hash({"physics": expanded["effective_physics"], "parameterizations": expanded["effective_parameterizations"]}),
        "environment": environment_identity(), "resource_estimate": resource_estimate(expanded),
        "cost": {"hardware": platform.processor() or platform.machine(), "precision": expanded["runtime_options"]["dtype"],
                 "cpu_ranks": 1, "timing_scope": "end_to_end", "compile_s": None, "integration_s": None, "io_s": None, "end_to_end_s": None},
    }
    write_json(directory / "expanded.json", expanded, create=True)
    write_json(directory / "manifest.json", manifest, create=True)
    return directory, manifest


def run_experiment(expanded, root, outputs, *, dry_run=False, evaluate=False, backend="cpu"):
    """CPU smoke by default; CUDA requires an explicit caller selection."""
    resources = expanded["resources"]
    validate_budget(resources, backend)
    directory, manifest = create_run(expanded, root, outputs)
    manifest["execution_backend"] = backend
    write_json(directory / "manifest.json", manifest)
    if dry_run:
        return manifest | {"run_directory": str(directory)}
    start = time.monotonic()
    manifest["execution_status"] = "running"
    manifest["started_at"] = datetime.now(timezone.utc).isoformat()
    write_json(directory / "manifest.json", manifest)
    try:
        data = data_identity(expanded["case"], root, require_files=True)
        manifest["data"] = data
        manifest["data_sha256"] = canonical_hash({key: value["observed_sha256"] for key, value in data.items()})
        write_json(directory / "manifest.json", manifest)
        environment = {**os.environ, "JAX_PLATFORMS": backend, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
        if backend == "cuda":
            environment.update(CUDA_VISIBLE_DEVICES="0", XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.40")
        manifest["worker_environment"] = {name: environment.get(name) for name in ("JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        command = [sys.executable, "-m", "zhenmode.execution.worker", "--run-directory", str(directory), "--root", str(Path(root).resolve())]
        manifest["command"] = command
        write_json(directory / "manifest.json", manifest)
        with (directory / "process.log").open("x", encoding="utf-8") as log:
            if backend == "cuda":
                remaining = max(0, resources["wall_seconds"] - (time.monotonic() - start))
                if not remaining:
                    raise subprocess.TimeoutExpired(command, resources["wall_seconds"])
                result, enforcement = run_cuda_worker(command, cwd=directory, env=environment, stdout=log, resources=resources | {"wall_seconds": remaining})
                manifest["resource_enforcement"] = enforcement
                manifest["exit_code"] = result.returncode
                if enforcement["stop_reason"]:
                    raise RuntimeError(f"CUDA worker stopped: {enforcement['stop_reason']}")
            else:
                result = subprocess.run(command, cwd=directory, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=resources["wall_seconds"], check=False)
        manifest["exit_code"] = result.returncode
        worker_report = directory / "worker-report.json"
        if not worker_report.is_file():
            raise RuntimeError(f"worker exited {result.returncode} without an execution report")
        report = json.loads(worker_report.read_text(encoding="utf-8"))
        allowed = {"executed_source_files", "result_path", "execution_hardware", "resource_enforcement", "verdict",
                   "requested_steps", "accepted_steps", "duration_complete", "forcing_provenance", "data_sha256",
                   "result_sha256", "result_source_identity", "physical_grid_sha256", "physical_grid_path", "physical_grid_artifact_sha256", "execution_grid_metadata", "timings", "device_memory_stats"}
        if set(report) - allowed:
            raise RuntimeError(f"worker report has unexpected fields: {sorted(set(report) - allowed)}")
        enforcement = manifest.get("resource_enforcement") if backend == "cuda" else None
        manifest.update(report)
        if enforcement is not None:
            manifest["resource_enforcement"] = enforcement
        if "timings" in report:
            manifest["cost"].update({name: report["timings"][name] for name in ("compile_s", "integration_s", "io_s")})
            manifest["cost"]["hardware"] = report["execution_hardware"]["device_kind"]
            manifest["cost"]["component_scope"] = report["timings"]["scope"]
        result_path = Path(report["result_path"])
        if not result_path.is_file() or file_hash(result_path) != report["result_sha256"]:
            raise RuntimeError("worker result artifact does not match recorded hash")
        manifest["outputs"] = [{"path": str(result_path), "sha256": report["result_sha256"], "role": "model_result"}]
        if "physical_grid_path" in report:
            grid_path = Path(report["physical_grid_path"])
            if not grid_path.is_file() or file_hash(grid_path) != report["physical_grid_artifact_sha256"]:
                raise RuntimeError("physical grid sidecar does not match recorded hash")
            manifest["outputs"].append({"path": str(grid_path), "sha256": report["physical_grid_artifact_sha256"], "role": "physical_grid"})
        # Production drift/skill FAIL with a finished integration is distinct
        # from abnormal termination. Evaluation still determines acceptance.
        complete = report.get("duration_complete") is True and result.returncode in (0, 1)
        manifest["execution_status"] = "completed" if complete else "failed"
        manifest["production_gate"] = {"verdict": report.get("verdict"), "status": "passed" if report.get("verdict") == "PASS" else "failed"}
        if not complete:
            manifest["failure_reason"] = f"production integration incomplete; worker exit {result.returncode}; inspect process.log"
    except subprocess.TimeoutExpired:
        manifest["execution_status"] = "failed"
        manifest["failure_reason"] = f"wall-clock limit exceeded ({resources['wall_seconds']} s); integration not verified complete"
    except KeyboardInterrupt:
        manifest["execution_status"] = "failed"
        manifest["failure_reason"] = "user/process interruption; integration not verified complete"
        raise
    except Exception as error:
        manifest["execution_status"] = "failed"
        manifest["failure_reason"] = f"{type(error).__name__}: {error}"
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["cost"]["end_to_end_s"] = time.monotonic() - start
        write_json(directory / "manifest.json", manifest)
    if evaluate and manifest["execution_status"] == "completed":
        try:
            from zhenmode.evaluation.pipeline import evaluate as evaluate_result

            protocol_path = Path(expanded["provenance"]["protocol_path"])
            if file_hash(protocol_path) != expanded["protocol_sha256"]:
                raise ConfigurationError("evaluation protocol changed after configuration freeze")
            evaluate_result(manifest["result_path"], protocol_path, directory / "manifest.json", directory / "evaluation")
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            manifest["evaluation"] = {"status": "completed", "acceptance_status": manifest["acceptance"]["status"]}
        except Exception as error:
            manifest["evaluation"] = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
        write_json(directory / "manifest.json", manifest)
    return manifest | {"run_directory": str(directory)}


def list_runs(outputs):
    records = []
    for path in sorted(Path(outputs).glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        records.append({name: manifest.get(name) for name in ("run_id", "case_id", "method", "preset_id", "experiment_id", "variant", "changes", "config_hash", "execution_status", "production_gate", "evaluation", "comparability", "acceptance")})
    return records
