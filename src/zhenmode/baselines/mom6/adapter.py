"""Pinned MOM6 ocean-only build and isolated tc1 end-to-end smoke workflow.

Run natively in Linux/WSL with the package installed. No system package or
credential modification is performed. Existing cluster/global artifacts remain
historical; tc1 is explicitly not the global shared-grid comparison.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import netCDF4
import numpy as np

from zhenmode.evaluation.protocols import digest, load_json
from zhenmode.provenance.sources import sha256_file as file_digest

RESOURCE_ROOT = Path(__file__).resolve().parent
PIN_SPEC = load_json(RESOURCE_ROOT / "pins.json")
PINS = {name: PIN_SPEC[name] for name in ("MOM6", "FMS", "CVMix", "GSW")}
CASE_SPEC = load_json(RESOURCE_ROOT / "cases/tc1.json")
CASE_ID = CASE_SPEC["id"]
INPUT_FILES = tuple(CASE_SPEC["input_files"])

LIMITATIONS = [
    "Official tc1 is a 10脳8脳8 Mercator forced model smoke, not the global ZhenMode case.",
    "Finite values, native conservation diagnostics and completion do not certify climate accuracy.",
    "No RMSE, common physical problem, industrial qualification or speed ranking is claimed.",
    "Only end-to-end process wall time is measured; compile/integration/IO split is not inferred.",
]


def _command(arguments, *, cwd=None, env=None, timeout=60):
    completed = subprocess.run([str(x) for x in arguments], cwd=cwd, env=env,
                               check=False, capture_output=True, text=True, timeout=timeout)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(map(str, arguments))}\n"
                           + completed.stderr[-3000:] + completed.stdout[-3000:])
    return completed.stdout.strip()


def paths(cache):
    cache = Path(cache).expanduser().resolve()
    legacy = (cache / "MOM6-f49a000").exists()
    return {"cache": cache,
            "MOM6": cache / ("MOM6-f49a000" if legacy else "sources/MOM6"),
            "FMS": cache / ("FMS-2023.03" if legacy else "sources/FMS"),
            "build_fms": cache / ("build-FMS-2023.03-r8" if legacy else "build/FMS"),
            "install_fms": cache / ("install-FMS-2023.03-r8" if legacy else "install/FMS"),
            "build_mom6": cache / ("build-MOM6-f49a000-r8" if legacy else "build/MOM6")}


def source_identity(directory, expected):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"pinned source missing: {directory}; run mom6 fetch")
    actual = _command(["git", "-C", directory, "rev-parse", "HEAD"])
    if actual != expected:
        raise ValueError(f"source commit mismatch: {directory}: {actual} != {expected}")
    dirty = _command(["git", "-C", directory, "status", "--porcelain", "--untracked-files=no"])
    if dirty:
        raise ValueError(f"tracked source is modified: {directory}\n{dirty}")
    names = _command(["git", "-C", directory, "ls-files", "-z"]).split("\0")
    tracked = {}
    for name in names:
        filename = directory / name
        if filename.is_file():
            tracked[name] = file_digest(filename)
    return {"commit": actual, "tracked_files": tracked, "tree_sha256": digest(tracked)}


def identities(cache):
    layout = paths(cache)
    result = {name: source_identity(layout[name], PINS[name]["commit"])
              for name in ("MOM6", "FMS")}
    for name in ("CVMix", "GSW"):
        result[name] = source_identity(layout["MOM6"] / PINS[name]["path"], PINS[name]["commit"])
    return result


def doctor(cache):
    tools = ("git", "cmake", "make", "autoreconf", "mpifort", "nf-config", "python3")
    result = {"platform": os.name, "cache": str(paths(cache)["cache"]), "pins": PINS,
              "tools": {name: shutil.which(name) for name in tools}, "limitations": LIMITATIONS}
    if os.name != "posix":
        result["ready"] = False
        result["error"] = "MOM6 runner requires Linux/WSL; install this package inside WSL and run there"
        return result
    try:
        result["source_identity"] = identities(cache)
        executable = paths(cache)["build_mom6"] / "MOM6"
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(f"MOM6 executable missing: {executable}; run mom6 build")
        result["executable"] = {"path": str(executable), "sha256": file_digest(executable)}
        result["binary_dependencies"] = binary_dependencies(cache, executable)
        result["build_provenance"] = build_provenance(cache, result["source_identity"],
                                                      result["executable"], result["binary_dependencies"])
        result["adapter_source_identity"] = adapter_identity()
        result["ready"] = True
    except (ValueError, OSError, RuntimeError) as error:
        result["ready"], result["error"] = False, str(error)
    result["environment"] = {"machine": os.uname().machine, "kernel": os.uname().release}
    for tool, args in (("mpifort", ["--version"]), ("nf-config", ["--all"])):
        if result["tools"][tool]:
            result["environment"][tool] = _command([tool, *args])
    return result


def build_provenance(cache, sources, executable, dependencies):
    receipt = paths(cache)["cache"] / "build_manifest.json"
    if not receipt.is_file():
        return {"status": "unverified", "reason": "No source-to-binary build receipt exists; existing executable and build flags were inspected only."}
    saved = load_json(receipt)
    commands = saved.get("build_commands")
    recorded_commands = (isinstance(commands, list) and bool(commands)
                         and all(isinstance(command, list) and command
                                 and all(isinstance(part, str) for part in command) for command in commands))
    matches = (saved.get("status") == "built" and recorded_commands
               and saved.get("sha256") == executable["sha256"] and saved.get("source_identity") == sources
               and saved.get("binary_dependencies") == dependencies)
    return {"status": "verified_recorded_build" if matches else "unverified",
            "receipt_path": str(receipt), "receipt_sha256": file_digest(receipt),
            "reason": "Build receipt binds pinned actual sources, commands, flags, binary and linked libraries."
            if matches else "Build receipt does not bind the currently inspected source/binary/dependencies."}


def adapter_identity():
    package = Path(__file__).resolve().parents[2]
    return {path.relative_to(package).as_posix(): file_digest(path)
            for path in sorted(package.rglob("*"))
            if path.is_file() and path.suffix in {".py", ".json"}}


def binary_dependencies(cache, executable):
    layout = paths(cache)
    libraries = {}
    for line in _command(["ldd", executable]).splitlines():
        match = re.search(r"(?:=>\s+)?(/\S+)", line)
        if match and Path(match[1]).is_file():
            libraries[str(Path(match[1]).resolve())] = file_digest(match[1])
    for library in (layout["install_fms"] / "lib").glob("*.a"):
        libraries[str(library.resolve())] = file_digest(library)
    build_files = {}
    for path in (layout["build_mom6"]/"Makefile", layout["build_mom6"]/"config.log",
                 layout["build_fms"]/"CMakeCache.txt", layout["MOM6"]/"ac/configure"):
        if path.is_file():
            build_files[str(path)] = {"sha256": file_digest(path)}
            if path.name == "Makefile":
                build_files[str(path)]["flags"] = [line for line in path.read_text().splitlines()
                                                    if re.match(r"(?:FC|FCFLAGS|LDFLAGS|LIBS)\s*=", line)]
    toolchain = {}
    for name in ("mpifort", "gfortran", "make", "cmake", "autoreconf", "nf-config"):
        binary = shutil.which(name)
        if binary:
            toolchain[name] = {"path": str(Path(binary).resolve()), "sha256": file_digest(binary),
                               "version": _command([name, "--version"])}
    return {"libraries": libraries, "build_files": build_files, "toolchain": toolchain,
            "status": "existing_cache_inspected_not_freshly_rebuilt"}


def fetch(cache):
    layout = paths(cache)
    layout["cache"].mkdir(parents=True, exist_ok=True)
    for name in ("MOM6", "FMS"):
        source = layout[name]
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            _command(["git", "clone", "--no-checkout", PINS[name]["url"], source], timeout=600)
            _command(["git", "-C", source, "checkout", "--detach", PINS[name]["commit"]])
        source_identity(source, PINS[name]["commit"])
    _command(["git", "-C", layout["MOM6"], "submodule", "update", "--init", "--recursive"],
             timeout=600)
    return {"pins": PINS, "source_identity": identities(cache), "status": "fetched"}


def build(cache, *, wall_seconds=1800):
    if os.name != "posix":
        raise ValueError("build inside Linux/WSL; no global software installation is performed")
    layout, identity = paths(cache), identities(cache)
    missing = [name for name in ("cmake", "make", "autoreconf", "mpifort", "nf-config")
               if shutil.which(name) is None]
    if missing:
        raise ValueError("missing external build dependencies: " + ", ".join(missing))
    executable = layout["build_mom6"] / "MOM6"
    if executable.is_file():
        dependencies = binary_dependencies(cache, executable)
        return {"status": "cached", "executable": str(executable),
                "sha256": file_digest(executable), "source_identity": identity,
                "build_provenance": build_provenance(cache, identity,
                    {"sha256": file_digest(executable)}, dependencies)}
    start, commands = time.monotonic(), []

    def execute(args, cwd=None, env=None):
        remaining = wall_seconds - (time.monotonic() - start)
        if remaining <= 0:
            raise TimeoutError("MOM6 build exceeded requested total wall budget")
        commands.append([str(x) for x in args])
        text = _command(args, cwd=cwd, env=env, timeout=remaining)
        layout["cache"].mkdir(parents=True, exist_ok=True)
        with (layout["cache"] / "build.log").open("a", encoding="utf-8") as log:
            log.write("\n" + " ".join(map(str, args)) + "\n" + text + "\n")

    execute(["cmake", "-S", layout["FMS"], "-B", layout["build_fms"],
             "-DCMAKE_Fortran_COMPILER=mpifort", "-DCMAKE_BUILD_TYPE=Release",
             "-D32BIT=OFF", "-DOPENMP=OFF", f"-DCMAKE_INSTALL_PREFIX={layout['install_fms']}"])
    execute(["cmake", "--build", layout["build_fms"], "--parallel", "1"])
    execute(["cmake", "--install", layout["build_fms"]])
    execute(["autoreconf"], cwd=layout["MOM6"] / "ac")
    layout["build_mom6"].mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, FC="mpifort",
               FCFLAGS=(f"-O2 -fallow-argument-mismatch -I{layout['install_fms']}/include_r8 "
                        + _command(["nf-config", "--fflags"]) + " -fdefault-real-8 -fdefault-double-8"),
               LDFLAGS=f"-L{layout['install_fms']}/lib",
               LIBS="-lFMS " + _command(["nf-config", "--flibs"]), OMP_NUM_THREADS="1")
    execute([layout["MOM6"] / "ac/configure", f"--srcdir={layout['MOM6']}"],
            cwd=layout["build_mom6"], env=env)
    execute(["make", "-j1"], cwd=layout["build_mom6"], env=env)
    result = {"schema_version": 1, "status": "built", "executable": str(executable),
              "sha256": file_digest(executable), "source_identity": identities(cache),
              "binary_dependencies": binary_dependencies(cache, executable),
              "build_commands": commands, "build_wall_s": time.monotonic()-start}
    (layout["cache"] / "build_manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def _write_manifest(directory, value):
    path = Path(directory) / "run.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def prepare(cache, *, run_root=None):
    check = doctor(cache)
    if not check["ready"]:
        raise ValueError(check["error"])
    layout = paths(cache)
    inputs = layout["MOM6"] / CASE_SPEC["upstream_case_path"]
    hashes = {name: file_digest(inputs / name) for name in INPUT_FILES}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + digest(hashes)[:12] + "-" + uuid.uuid4().hex[:8]
    root = Path(run_root).expanduser().resolve() if run_root else layout["cache"] / "runs"
    directory = root / run_id
    directory.mkdir(parents=True, exist_ok=False)
    for name in INPUT_FILES:
        shutil.copyfile(inputs / name, directory / name)
    for name in ("INPUT", "RESTART"):
        (directory / name).mkdir()
    manifest = {"schema_version": 1, "run_id": run_id, "case_id": CASE_ID, "method": "MOM6",
                "preset_id": "official-tc1-pinned", "experiment_id": "mom6-tc1-smoke",
                "variant": "upstream_unmodified", "changes": [], "config_hash": digest(hashes),
                "physical_problem_sha256": digest({"case": CASE_ID, "inputs": hashes}),
                "data_sha256": digest(hashes), "inputs": hashes,
                "source_identity": check["source_identity"], "executable": check["executable"],
                "binary_dependencies": check["binary_dependencies"],
                "build_provenance": check["build_provenance"],
                "adapter_source_identity": check["adapter_source_identity"],
                "environment": check["environment"], "cache": str(layout["cache"]),
                "execution_status": "proposed",
                "comparability": "not_comparable_to_global_case", "acceptance": "not_evaluated",
                "evaluation_protocol": "mom6-tc1-smoke-v1", "expected_days": 0.25,
                "expected_steps": 24, "limitations": LIMITATIONS}
    _write_manifest(directory, manifest)
    return {"run_dir": str(directory), "run_id": run_id, "execution_status": "proposed"}


def _process_group_rss(group):
    total = 0
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            if int(stat[2]) != group:
                continue
            for line in (proc / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            continue
    return total


def _output_bytes(directory):
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def run(directory, *, wall_seconds=180, memory_bytes=4*1024**3, output_bytes=128*1024**2):
    if os.name != "posix":
        raise ValueError("run MOM6 inside Linux/WSL for enforced CPU/process-group resource limits")
    directory = Path(directory).resolve()
    manifest = load_json(directory / "run.json")
    if manifest["execution_status"] != "proposed":
        raise ValueError("each prepared run executes once; prepare a new run for repeats")
    if not 0 < wall_seconds <= 180 or not 0 < memory_bytes <= 4*1024**3:
        raise ValueError("bounded smoke requires <=180 seconds and <=4 GiB; longer experiments need a separate protocol")
    for name, expected in manifest["inputs"].items():
        if file_digest(directory / name) != expected:
            raise ValueError(f"prepared input changed: {name}")
    executable = Path(manifest["executable"]["path"])
    if file_digest(executable) != manifest["executable"]["sha256"]:
        raise ValueError("executable changed after preparation")
    if "binary_dependencies" in manifest:
        for filename, expected in manifest["binary_dependencies"]["libraries"].items():
            if file_digest(filename) != expected:
                raise ValueError(f"linked/static dependency changed: {filename}")
    if "adapter_source_identity" in manifest and adapter_identity() != manifest["adapter_source_identity"]:
        raise ValueError("Python orchestration package changed after preparation")
    # Verify every tracked execution source and linked dependency, not a facade file.
    for name, identity in manifest["source_identity"].items():
        layout = paths(manifest["cache"])
        source = layout[name] if name in layout else layout["MOM6"] / PINS[name]["path"]
        if source_identity(source, identity["commit"]) != identity:
            raise ValueError(f"execution source changed after preparation: {name}")
    allowed = os.sched_getaffinity(0)
    core = min(allowed)

    def limits():
        import resource
        os.sched_setaffinity(0, {core})
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (output_bytes, output_bytes))

    manifest.update(execution_status="running", resources={"cpu": 1, "ranks": 1,
                     "wall_seconds": wall_seconds, "memory_bytes": memory_bytes,
                     "output_bytes": output_bytes, "cpu_affinity": [core]})
    _write_manifest(directory, manifest)
    start, peak, failure = time.monotonic(), 0, None
    process = None
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    try:
        with (directory / "launch.log").open("xb") as log:
            process = subprocess.Popen([str(executable)], cwd=directory, env=env,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, preexec_fn=limits)
            while process.poll() is None:
                peak = max(peak, _process_group_rss(process.pid))
                if time.monotonic()-start > wall_seconds:
                    failure = "wall_time_limit"
                elif peak > memory_bytes:
                    failure = "aggregate_memory_limit"
                elif _output_bytes(directory) > output_bytes:
                    failure = "aggregate_output_limit"
                if failure:
                    os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(0.05)
            code = process.wait()
        if code != 0 and failure is None:
            failure = f"nonzero_exit_{code}"
        if failure is None:
            manifest["native_output_receipt"] = {
                str(filename.relative_to(directory)): file_digest(filename)
                for filename in sorted(directory.rglob("*.nc")) if filename.is_file()}
            required_outputs = ({"ocean.stats.nc"} if manifest["case_id"] == CASE_ID else
                                {"ocean.stats.nc", "prog.nc", "ocean_geometry.nc"})
            if not required_outputs <= set(manifest["native_output_receipt"]):
                failure = "missing_native_output_at_launch_completion"
        manifest.update(execution_status="failed" if failure else "completed", exit_code=code,
                        failure=failure, aggregate_peak_bytes=peak,
                        cost={"hardware": manifest["environment"]["machine"], "precision": "fp64",
                              "cpu_ranks": 1, "timing_scope": "native_process_end_to_end",
                              "compile_s": None, "integration_s": None, "io_s": None,
                              "end_to_end_s": time.monotonic()-start})
    except BaseException as error:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        manifest.update(execution_status="failed", failure=type(error).__name__ + ": " + str(error))
        _write_manifest(directory, manifest)
        raise
    _write_manifest(directory, manifest)
    return {"run_dir": str(directory), "run_id": manifest["run_id"],
            "execution_status": manifest["execution_status"], "failure": failure,
            "cost": manifest["cost"], "aggregate_peak_bytes": peak}


def convert(directory):
    directory = Path(directory).resolve()
    manifest = load_json(directory / "run.json")
    if manifest["execution_status"] != "completed":
        raise ValueError("conversion requires a completed native run")
    launch_outputs = manifest.get("native_output_receipt")
    if not isinstance(launch_outputs, dict) or not launch_outputs:
        raise ValueError("native output has no launch-completion hash receipt; historical data need separate evidence import")
    for name, expected in launch_outputs.items():
        if file_digest(directory/name) != expected:
            raise ValueError(f"native launch output changed before conversion: {name}")
    if manifest["case_id"] != CASE_ID:
        prog, geometry = directory/"prog.nc", directory/"ocean_geometry.nc"
        if "result_sha256" in manifest:
            raise FileExistsError("native output conversion already registered")
        for filename in (prog, geometry):
            if not filename.is_file():
                raise FileNotFoundError(f"native shared-grid output missing: {filename}")
        manifest.update(result_path=str(prog), result_sha256=file_digest(prog),
                        geometry_path=str(geometry), geometry_sha256=file_digest(geometry),
                        outputs=[{"path": str(prog), "sha256": file_digest(prog), "role": "model_result"},
                                 {"path": str(geometry), "sha256": file_digest(geometry), "role": "native_geometry"}])
        _write_manifest(directory, manifest)
        return {"result_path": str(prog), "result_sha256": manifest["result_sha256"],
                "geometry_path": str(geometry), "conversion": "native_names_mapped_at_evaluation_no_regridding"}
    output = directory / "native_diagnostics.npz"
    if output.exists():
        raise FileExistsError(f"conversion already exists: {output}")
    stats = directory / "ocean.stats.nc"
    with netCDF4.Dataset(stats) as ds:
        required = {"Time", "En", "Mass", "Heat", "Salt", "max_CFL_trans", "max_CFL_lin", "Ntrunc"}
        if not required.issubset(ds.variables):
            raise ValueError(f"native statistics missing: {sorted(required-set(ds.variables))}")
        arrays = {name: np.asarray(np.ma.filled(ds[name][:], np.nan), dtype=float)
                  for name in required}
        time_units = getattr(ds["Time"], "units", "")
        # The pinned statistics use model days, not CF calendar time. No guessed units.
        if time_units != "days":
            raise ValueError(f"unexpected native time units: {time_units}")
        units = {name: getattr(ds[name], "units", None) for name in required}
        expected_units = {"Time": "days", "En": "Joules", "Mass": "kg", "Heat": "Joules", "Salt": "kg",
                          "max_CFL_trans": "Nondim", "max_CFL_lin": "Nondim", "Ntrunc": "Nondim"}
        if units != expected_units:
            raise ValueError("native diagnostic units differ from pinned tc1 interface")
    receipt = {"source_file": str(stats), "source_sha256": file_digest(stats),
               "run_id": manifest["run_id"], "case_id": CASE_ID, "units": units,
               "time_units": time_units, "sampling": "native_global_diagnostics_saved_records"}
    with output.open("xb") as stream:
        np.savez_compressed(stream, **arrays, metadata=np.array(json.dumps(receipt)))
    artifact = {"output": str(output), "sha256": file_digest(output), "receipt": receipt}
    manifest["converted_output"] = artifact
    _write_manifest(directory, manifest)
    return artifact


def evaluate(directory):
    from zhenmode.evaluation.metrics import _relative_drift

    directory = Path(directory).resolve()
    manifest = load_json(directory / "run.json")
    if manifest["case_id"] != CASE_ID:
        from zhenmode.evaluation.pipeline import evaluate as evaluate_sst
        if file_digest(manifest["geometry_path"]) != manifest["geometry_sha256"]:
            raise ValueError("actual native geometry changed after conversion")
        if file_digest(manifest["reference_npz"]) != manifest["reference_npz_sha256"]:
            raise ValueError("shared initial reference changed after preparation")
        return evaluate_sst(manifest["result_path"], manifest["protocol_path"], directory/"run.json",
                            directory/"evaluation", format="mom6", reference=manifest["reference_npz"],
                            geometry=manifest["geometry_path"])
    report_path = directory / "report.json"
    if report_path.exists():
        raise FileExistsError(f"report already exists: {report_path}")
    if manifest["execution_status"] != "completed":
        raise ValueError("evaluation requires completed run; failed execution is preserved in run.json")
    converted = directory / "native_diagnostics.npz"
    if file_digest(converted) != manifest.get("converted_output", {}).get("sha256"):
        raise ValueError("converted output changed or not registered by converter")
    with np.load(converted, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files if name != "metadata"}
        receipt = json.loads(str(saved["metadata"]))
    if receipt["run_id"] != manifest["run_id"] or receipt["case_id"] != CASE_ID:
        raise ValueError("converted output identity differs from run")
    if file_digest(receipt["source_file"]) != receipt["source_sha256"]:
        raise ValueError("native statistics changed after conversion")
    days = arrays["Time"]
    if days.ndim != 1 or not days.size or any(value.shape != days.shape for value in arrays.values()):
        raise ValueError("all native smoke diagnostic timelines must be nonempty and match Time shape")
    checks = {"finite_diagnostics": all(np.all(np.isfinite(value)) for value in arrays.values()),
              "ordered_times": days.ndim == 1 and days.size >= 2 and bool(np.all(np.diff(days) > 0)),
              "duration_complete": bool(np.isclose(days[-1], manifest["expected_days"], rtol=0, atol=1e-12)),
              "positive_mass": bool(np.all(arrays["Mass"] > 0)),
              "cfl_below_one": all(bool(np.all((arrays[name] >= 0) & (arrays[name] < 1)))
                                    for name in ("max_CFL_trans", "max_CFL_lin")),
              "no_velocity_truncations": bool(np.all(arrays["Ntrunc"] == 0))}
    report = {"schema_version": 1, "protocol": "mom6-tc1-smoke-v1", "case_id": CASE_ID,
              "method": "MOM6", "run_id": manifest["run_id"], "execution_status": "completed",
              "numerical": {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
                            "mass_drift_percent": _relative_drift(arrays["Mass"]),
                            "heat_drift_percent": _relative_drift(arrays["Heat"]),
                            "salt_drift_percent": _relative_drift(arrays["Salt"]),
                            "scope": "pinned_native_short_smoke_not_independent_budget_closure"},
              "effect": {"status": "not_evaluated", "rmse": None}, "cost": manifest["cost"],
              "comparability": "not_comparable_to_global_case", "ranking": "not_performed",
              "acceptance": {"status": "passed" if all(checks.values()) else "failed",
                             "scope": "finite_completed_tc1_smoke_only"},
              "source_identity": manifest["source_identity"], "executable": manifest["executable"],
              "binary_dependencies": manifest.get("binary_dependencies", {"status": "not_recorded_at_launch"}),
              "build_provenance": manifest.get("build_provenance", {"status": "unverified", "reason": "not_recorded_at_launch"}),
              "adapter_source_identity": manifest.get("adapter_source_identity", {"status": "not_recorded_at_launch"}),
              "scoring_source_identity": adapter_identity(),
              "inputs": manifest["inputs"], "output": receipt, "limitations": LIMITATIONS,
              "industrial_qualified": False}
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (directory / "report.md").write_text(
        f"# MOM6 tc1 {manifest['run_id']}\n\nExecution: completed; smoke acceptance: "
        f"{report['acceptance']['status']}.\n\nChecks:\n\n" +
        "\n".join(f"- {name}: {value}" for name, value in checks.items()) +
        "\n\nLimitations:\n\n" + "\n".join("- " + item for item in LIMITATIONS)+"\n",
        encoding="utf-8")
    manifest["acceptance"] = report["acceptance"]
    _write_manifest(directory, manifest)
    return report


def native_parameters(directory):
    """Small strict reader for the declared global parameter subset, not a MOM parser.

    Complex native syntax is retained in the launch files and resolved docs. This
    preflight never claims equivalence for unparsed native physics.
    """
    result = {}
    for filename in ("MOM_input", "MOM_override"):
        within_file = set()
        for line in (Path(directory)/filename).read_text().splitlines():
            line = line.split("!", 1)[0].strip()
            if line.startswith("#override"):
                line = line[len("#override"):].strip()
            matched = re.fullmatch(r"([A-Z][A-Z0-9_]*)\s*=\s*(.+)", line)
            if not matched:
                continue
            key, raw = matched.groups()
            if key in within_file:
                raise ValueError(f"ambiguous repeated native parameter in {filename}: {key}")
            within_file.add(key)
            raw = raw.strip()
            if raw.lower() in {"true", "false"}:
                value = raw.lower() == "true"
            elif raw.startswith('"') and raw.endswith('"'):
                value = raw[1:-1]
            else:
                try:
                    value = float(raw.replace("D", "e").replace("d", "e"))
                except ValueError:
                    value = raw
            result[key] = value
    return result


def prepare_native(cache, *, case_path, preset_path, native_dir, reference_npz,
                   root, run_root=None):
    """Bind existing shared global native inputs to a common case and preset.

    This is deliberately a checked external-input adapter. It does not recreate
    missing historical native files or silently substitute a climate forcing.
    """
    from zhenmode.execution.resolve import canonical_hash, read_case
    from zhenmode.execution.schema import load_document

    case, _ = read_case(case_path)
    preset = load_document(preset_path)
    allowed = {"schema_version", "kind", "id", "name_zh", "method", "case_id", "case",
               "protocol", "native_options", "required_inputs", "discretization", "limitations"}
    if set(preset) != allowed or preset["method"] != "MOM6" or preset["case_id"] != case["id"]:
        raise ValueError("MOM6 preset fields/method/common case identity differ")
    if case["id"] not in {"global-050deg-wind-only-30d", "global-050deg-restoring-30d"}:
        raise ValueError("native global adapter supports the two declared existing 0.5-degree cases")
    native_dir = Path(native_dir).resolve()
    parameters = native_parameters(native_dir)
    dt = parameters.get("DT")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("actual native DT must be explicit, finite positive seconds")
    for name, value in preset["native_options"].items():
        actual = parameters.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            matches = actual is not None and isinstance(actual, (int, float)) and np.isclose(actual, value, rtol=0, atol=1e-8)
        else:
            matches = actual == value
        if not matches:
            raise ValueError(f"native input differs from declared preset: {name}: {actual!r} != {value!r}")
    bindings = {"bathymetry": ("TOPO_FILE", "TOPO_VARNAME", "depth"),
                "temperature": ("TEMP_Z_INIT_FILE" if "TEMP_Z_INIT_FILE" in parameters else "TEMP_SALT_Z_INIT_FILE", "Z_INIT_FILE_PTEMP_VAR", "ptemp"),
                "salinity": ("SALT_Z_INIT_FILE" if "SALT_Z_INIT_FILE" in parameters else "TEMP_SALT_Z_INIT_FILE", "Z_INIT_FILE_SALT_VAR", "salt"),
                "wind_x": ("WIND_FILE", "WINDSTRESS_X_VAR", "STRESS_X"),
                "wind_y": ("WIND_FILE", "WINDSTRESS_Y_VAR", "STRESS_Y"),
                "sst_restore": ("SSTRESTORE_FILE", "SST_RESTORE_VAR", "SST"),
                "sss_restore": ("SALINITYRESTORE_FILE", "SSS_RESTORE_VAR", "SSS")}
    for item in preset["required_inputs"]:
        file_parameter, variable_parameter, default_variable = bindings[item["role"]]
        native_reference = parameters.get(file_parameter)
        if not isinstance(native_reference, str):
            raise ValueError(f"native input role {item['role']} requires active parameter {file_parameter}")
        selected = Path(native_reference) if "/" in native_reference else Path(parameters["INPUTDIR"])/native_reference
        if selected.is_absolute() or ".." in selected.parts or selected.as_posix() != item["path"]:
            raise ValueError(f"actual native {file_parameter} differs from declared input {item['path']}")
        if parameters.get(variable_parameter, default_variable) != item["variable"]:
            raise ValueError(f"actual native variable differs for role {item['role']}")
    with np.load(reference_npz, allow_pickle=False) as ref:
        required = {"T_init", "S_init", "wet_mask", "lat", "lon", "depth"}
        if not required.issubset(ref.files):
            raise ValueError(f"shared reference missing physical grid/state arrays: {sorted(required-set(ref.files))}")
        lat, lon, raw_wet = ref["lat"], ref["lon"], np.asarray(ref["wet_mask"])
        if not np.all(np.isfinite(raw_wet)) or not np.all((raw_wet == 0) | (raw_wet == 1)):
            raise ValueError("shared reference wet mask must be finite binary values")
        wet = raw_wet.astype(bool)
        if wet.shape != (720, 260) or lat.size != 260 or lon.size != 720:
            raise ValueError("declared 0.5-degree global shared grid must be 720脳260")
        physical_grid = {"depth": digest(np.asarray(ref["depth"]).tolist()),
                         "wet_mask": digest(wet.tolist()), "lat": digest(lat.tolist()), "lon": digest(lon.tolist())}
        for item in preset["required_inputs"]:
            filename = native_dir / item["path"]
            if not filename.is_file():
                raise FileNotFoundError(f"native input missing for {item['role']}: {filename}")
            with netCDF4.Dataset(filename) as ds:
                variable = item["variable"]
                if variable not in ds.variables:
                    raise ValueError(f"missing native {variable}: {filename}")
                units = getattr(ds[variable], "units", "")
                allowed_units = {"bathymetry": {"m", "meters", "meter"},
                                 "temperature": {"degC", "degrees_Celsius", "degrees_C", "Celsius"},
                                 "salinity": {"psu", "PSU", "1e-3"},
                                 "wind_x": {"N m-2", "Pa"}, "wind_y": {"N m-2", "Pa"},
                                 "sst_restore": {"degC", "degrees_Celsius", "Celsius"},
                                 "sss_restore": {"psu", "PSU", "1e-3"}}
                if units not in allowed_units[item["role"]]:
                    raise ValueError(f"native input unit mismatch for {item['role']}: {units!r}")
                for dimension, expected in zip(ds[variable].dimensions[-2:], (lat, lon), strict=True):
                    if dimension not in ds.variables or ds[dimension].shape != expected.shape or not np.allclose(ds[dimension][:], expected, rtol=0, atol=1e-6):
                        raise ValueError(f"native input coordinate/dimension mismatch: {filename}: {dimension}")
                if item["role"] == "bathymetry":
                    depth = np.asarray(ds[variable][:], dtype=float).T
                    if depth.shape != wet.shape or not np.array_equal(depth > 0, wet):
                        raise ValueError("native bathymetry wet mask differs from shared reference")
                    if not np.allclose(depth[wet], np.asarray(ref["depth"])[wet], rtol=0, atol=1e-6):
                        raise ValueError("actual smoothed/min-depth bathymetry differs from shared reference")
                elif item["role"] in {"temperature", "salinity"}:
                    state = np.asarray(ds[variable][:], dtype=float)
                    expected = ref["T_init" if item["role"] == "temperature" else "S_init"]
                    if state.shape == (expected.shape[2], expected.shape[1], expected.shape[0]):
                        state = state.transpose(2, 1, 0)
                    if state.shape != expected.shape or not np.allclose(state[wet], expected[wet], rtol=0, atol=1e-6):
                        raise ValueError("native initial T/S does not match shared reference (no implicit vertical remap)")
    # Check real raw data availability/version identity separately from native derived input bytes.
    data = {}
    for item in case["data"]:
        filename = Path(root)/item["path"]
        actual = file_digest(filename)
        if item["sha256"] is not None and actual != item["sha256"]:
            raise ValueError(f"common case raw data checksum mismatch: {filename}")
        data[item["role"]] = {"sha256": actual, "version": item["version"]}
    prepared = prepare(cache, run_root=run_root)
    destination = Path(prepared["run_dir"])
    manifest = load_json(destination/"run.json")
    manifest.update(case_id=case["id"], execution_status="failed", failure="native_preparation_not_complete")
    _write_manifest(destination, manifest)
    # Newly-created tc1 scaffold is now replaced by declared global native inputs.
    for name in INPUT_FILES:
        shutil.copyfile(native_dir/name, destination/name)
    for filename in (native_dir/"INPUT").rglob("*"):
        if filename.is_file():
            target = destination/"INPUT"/filename.relative_to(native_dir/"INPUT")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(filename, target)
    files = {str(filename.relative_to(destination)): file_digest(filename)
             for filename in destination.rglob("*") if filename.is_file() and filename.name != "run.json"}
    protocol_path = (Path(root)/"protocols"/(preset["protocol"]+".json")).resolve()
    from zhenmode.evaluation.protocols import load_protocol
    protocol = load_protocol(protocol_path)
    if protocol["case_id"] != case["id"]:
        raise ValueError("native preset's evaluation protocol differs from common case")
    manifest.update(case_id=case["id"], preset_id=preset["id"], experiment_id=preset["id"],
                    execution_status="proposed", failure=None,
                    variant="shared_native_input_adaptation", case=case, case_sha256=file_digest(case_path),
                    preset_sha256=file_digest(preset_path), config_hash=digest(files), inputs=files,
                    native_parameters=parameters, physical_grid_sha256=digest(physical_grid),
                    physical_problem_sha256=canonical_hash({k: case[k] for k in ("domain", "grid", "initial", "forcing", "duration", "output_interval")}),
                    effective_physics_sha256=None,
                    data_sha256=digest(data), raw_data=data, reference_npz=str(Path(reference_npz).resolve()),
                    reference_npz_sha256=file_digest(reference_npz), expected_days=30.0,
                    expected_steps=int(30*86400/float(parameters["DT"])),
                    evaluation_protocol=preset["protocol"], protocol_path=str(protocol_path),
                    protocol_file_sha256=file_digest(protocol_path), protocol_content_sha256=digest(protocol),
                    comparability="limited_native_physics_not_fully_matched",
                    limitations=preset["limitations"] + ["Native T/S and actual bathymetry match checked; wind transformation, vertical coordinate and full physics equivalence remain reviewer-owned."])
    _write_manifest(destination, manifest)
    return {"run_dir": str(destination), "run_id": manifest["run_id"], "case_id": case["id"],
            "execution_status": "proposed", "comparability": manifest["comparability"],
            "estimated_resources": {"historical_4_rank_wall_s": 3000, "default_guard_wall_s": 180,
                                    "cpu": 1, "memory_bytes": 4*1024**3},
            "limitations": manifest["limitations"]}


def prepare_wave_input(contract_path, out_dir):
    """Prepare exact MOM cell-mean interfaces for the existing independent v0 contract.

    This supplies initialization, not a solver, run receipt, or approval to run.
    Native option support, zero-process resolved parameters and the independent
    research scorer remain a separate preflight/acceptance contract.
    """
    contract = load_json(contract_path)
    required = {"schema": "standing-wave-v0", "ny": 8, "nz": 4, "H_m": 100.0,
                "gravity": 9.81, "rho0": 1025.0, "f": 0.0, "T_C": 15.0, "S_psu": 35.0,
                "period_s": 32000.0, "Ly_m": 100000.0}
    if any(contract.get(name) != value for name, value in required.items()):
        raise ValueError("input differs from frozen standing-wave-v0 physical contract")
    nx, ny = contract.get("nx"), contract["ny"]
    if nx not in {64, 128, 256} or contract.get("amplitude_m") not in {.01, .005}:
        raise ValueError("unsupported standing wave resolution/amplitude")
    length = float(contract["Lx_m"])
    if not np.isclose(length, 32000*np.sqrt(9.81*100), rtol=0, atol=1e-8):
        raise ValueError("standing wave physical Lx differs")
    if contract.get("native_sampling", {}).get("MOM6") != {"eta": "cell_mean", "u": "node", "layout": "cgrid"}:
        raise ValueError("MOM6 requires exact cell-mean eta and native C-grid point velocities")
    destination = Path(out_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    x, y = (np.arange(nx)+.5)*length/nx, (np.arange(ny)+.5)*contract["Ly_m"]/ny
    surface = contract["amplitude_m"]*np.cos(2*np.pi*x/length)*np.sinc(1/nx)
    interfaces = np.broadcast_to(np.array([0., -100/6, -50., -500/6, -100.])[:, None, None],
                                 (5, ny, nx)).copy()
    interfaces[0] = surface[None, :]
    filename = destination/"standing_wave_initial.nc"
    with netCDF4.Dataset(filename, "w") as ds:
        for name, size in (("x", nx), ("y", ny), ("Layer", 4), ("Interface", 5)):
            ds.createDimension(name, size)
        for name, values in (("x", x), ("y", y)):
            coordinate = ds.createVariable(name, "f8", (name,))
            coordinate.units = "m"
            coordinate.cartesian_axis = name.upper()
            coordinate[:] = values
        height = ds.createVariable("eta", "f8", ("Interface", "y", "x"))
        height.units, height.positive = "m", "up"
        height[:] = interfaces
        for name, value, unit in (("ptemp", 15., "degC"), ("salt", 35., "psu")):
            variable = ds.createVariable(name, "f8", ("Layer", "y", "x"))
            variable.units = unit
            variable[:] = value
        ds.contract_sha256 = digest(contract)
        ds.eta_sampling = "exact_cell_mean"
    receipt = {"schema_version": 1, "contract_sha256": digest(contract),
               "source_contract_sha256": file_digest(contract_path), "input_path": str(filename),
               "input_sha256": file_digest(filename), "execution_status": "proposed",
               "sampling": {"eta": "cell_mean", "u": "node", "layout": "cgrid"},
               "native_interface": {"THICKNESS_CONFIG": "file", "THICKNESS_FILE": filename.name,
                                    "INTERFACE_IC_VAR": "eta"},
               "limitations": ["Only frozen initial interfaces/T/S are prepared; no full-model run or scoring claimed.",
                               "Pinned native resolved zero-process/time-scheme and exact full output schema require independent preflight before execution."]}
    (destination/"preparation.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return receipt
