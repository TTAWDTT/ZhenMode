"""Persistent single-rank native SIS2 exchange; physics stays in the pinned build.

Run this client inside the existing Linux resource supervisor. Arrays use
float64 Fortran order, with explicit category axes. The native process keeps
its full ice/snow/dynamics state between requests; this is not a restart format.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from zhenmode.provenance.sources import sha256_file

OCEAN_FIELDS = ("u", "v", "t", "s", "frazil", "sea_level")
LAND_FIELDS = ("runoff", "calving", "runoff_hflx", "calving_hflx")
AIR_FIELDS = tuple(
    "u_flux v_flux u_star t_flux q_flux lw_flux sw_flux_vis_dir sw_flux_vis_dif "
    "sw_flux_nir_dir sw_flux_nir_dif sw_down_vis_dir sw_down_vis_dif "
    "sw_down_nir_dir sw_down_nir_dif lprec fprec dhdt dedt drdt coszen p".split()
)
FLUX_FIELDS = tuple(
    "flux_u flux_v flux_t flux_q flux_lw flux_lh sw_vis_dir sw_vis_dif "
    "sw_nir_dir sw_nir_dif lprec fprec runoff calving runoff_hflx "
    "calving_hflx flux_salt mi p_surf enth_mass_in_ocn enth_mass_out_ocn "
    "enth_mass_in_atm enth_mass_out_atm frazil_left".split()
)
SURFACE_FIELDS = tuple(
    "part_size t_surf u_surf v_surf rough_mom rough_heat rough_moist "
    "albedo_vis_dir albedo_vis_dif albedo_nir_dir albedo_nir_dif".split()
)


def compile_sis2_bridge(coupled_build, output):
    """Link the packaged driver against recorded, unchanged native build objects."""
    from zhenmode.baselines.mom6.coupled_sources import source_tree
    from zhenmode.baselines.mom6.omip2 import _build_artifacts, _linked_libraries
    from zhenmode.evaluation.protocols import digest
    from zhenmode.execution.resources import run_process_group
    from zhenmode.provenance.sources import load_json

    root, output = Path(coupled_build).resolve(), Path(output).resolve()
    recipe = load_json(root / "recipe.json")
    sources = load_json(root / "executed-sources.json")
    artifacts = load_json(root / "artifact-checkpoint.json")["artifacts"]
    if (
        recipe["target"] != "ice_ocean_SIS2"
        or source_tree(root / "sources") != sources
        or digest(sources) != recipe["executed_source_tree_sha256"]
        or _build_artifacts(root / "build") != artifacts
    ):
        raise ValueError("native coupled source/build checkpoint identity mismatch")
    output.mkdir(parents=True, exist_ok=False)
    build, driver = root / "build", Path(__file__).with_name("sis2_bridge.f90")
    driver_id = sha256_file(driver)
    objects = sorted(
        p for p in (build / "ice_ocean_SIS2").glob("*.o") if p.name != "coupler_main.o"
    )
    if not objects:
        raise ValueError("native SIS2 build has no reusable objects")
    command = ["mpif90", "-O1", "-fdefault-real-8", "-fdefault-double-8", "-ffree-line-length-none"]
    for directory in ("ice_ocean_SIS2", "fms", "icebergs", "ice_param", "atmos_null", "land_null"):
        command += ["-I" + str(build / directory), "-L" + str(build / directory)]
    program = output / "sis2_bridge"
    command += [
        str(driver),
        *map(str, objects),
        "-licebergs",
        "-lice_param",
        "-latmos_null",
        "-lland_null",
        "-lFMS",
        "-lnetcdff",
        "-lnetcdf",
        "-o",
        str(program),
    ]
    with (output / "compile.log").open("x") as log:
        completed, resources = run_process_group(
            command,
            cwd=output,
            env=dict(os.environ, OMP_NUM_THREADS="1"),
            stdout=log,
            resources={"cpu": 1, "memory_mib": 4096, "wall_seconds": 180},
        )
    unchanged = (
        sha256_file(driver) == driver_id
        and _build_artifacts(build) == artifacts
        and source_tree(root / "sources") == sources
    )
    receipt = {
        "command": command,
        "returncode": completed.returncode,
        "resources": resources,
        "native_build_recipe_sha256": sha256_file(root / "recipe.json"),
        "native_artifact_checkpoint_sha256": sha256_file(root / "artifact-checkpoint.json"),
        "driver_sha256": driver_id,
        "inputs_unchanged": unchanged,
        "program_sha256": sha256_file(program) if program.exists() else None,
        "native_build_recipe": recipe,
        "full_case_qualification": False,
    }
    if completed.returncode == 0:
        receipt["linked_libraries"] = _linked_libraries(program)
    (output / "bridge.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf8")
    if completed.returncode != 0 or not unchanged:
        raise RuntimeError("native SIS2 bridge linking failed or inputs changed")
    return receipt


def decode_reply(path, shape, *, fluxes):
    """Decode exact payload length; salt flux is kg/m²/s OUT of ocean.

    Net LW and SW are INTO ocean; sensible/latent and evaporation are OUT.
    These signs follow the actual SIS2 setters and MOM6 FMS-cap receiver.
    Public comments alone are insufficient (the LW comment is inconsistent).
    """
    nx, ny, nc = shape
    fields = [(n, (nx, ny)) for n in FLUX_FIELDS] if fluxes else []
    fields += [(n, shape) for n in SURFACE_FIELDS]
    fields += [(n, (nx, ny)) for n in ("s_surf", "area")]
    count = sum(np.prod(s) for _, s in fields) + 6
    path = Path(path)
    if path.stat().st_size != count * 8:
        raise ValueError("native SIS2 reply size mismatch")
    data = np.fromfile(path, dtype="<f8")
    if not np.isfinite(data).all():
        raise ValueError("nonfinite native SIS2 reply")
    result, offset = {}, 0
    for name, dims in fields:
        length = int(np.prod(dims))
        result[name] = data[offset : offset + length].reshape(dims, order="F").copy()
        offset += length
    result["stocks"] = data[-6:-3].copy()  # water kg, heat J, salt kg, native stock routine
    result["thermo_constants"] = data[-3:].copy()  # fusion J/kg, water Cp J/kg/K, ice kg/m³
    wet = result["area"] > 0
    fractions = result["part_size"]
    if np.any(result["area"] < 0) or np.any(fractions < -1e-13):
        raise ValueError("invalid native ice area or category fraction")
    if np.any(np.abs(fractions.sum(-1)[wet] - 1) > 1e-12):
        raise ValueError("native ice categories do not cover wet cell")
    return result


class NativeSIS2:
    """One native process, explicit field validation, retained exchange payloads.

    The caller supplies an exclusive prepared case and SHA-256 of the linked
    bridge. Construction and every reply have a timeout; failures terminate
    the native child. A surrounding supervisor bounds aggregate RSS and CPU.
    """

    def __init__(
        self,
        program,
        case,
        *,
        program_sha256,
        dt_seconds,
        steps,
        timeout=30,
        start="1958-01-01T00:00:00",
        elapsed_seconds=0,
    ):
        if os.name != "posix":
            raise ValueError("native SIS2 bridge requires Linux/WSL")
        if any(type(v) is not int or v <= 0 for v in (dt_seconds, steps)):
            raise ValueError("native ice interval and step limit must be positive integers")
        if not np.isfinite(timeout) or not 0 < timeout <= 180:
            raise ValueError("native reply timeout must be 0..180 seconds")
        timestamp = datetime.fromisoformat(start)
        if timestamp.isoformat(timespec="seconds") != start or timestamp.tzinfo is not None:
            raise ValueError("native start must be Gregorian YYYY-MM-DDTHH:MM:SS")
        if type(elapsed_seconds) is not int or not 0 <= elapsed_seconds < 2**31:
            raise ValueError("native elapsed clock must fit a nonnegative 32-bit interval")
        self.program, self.case = Path(program).resolve(), Path(case).resolve()
        if sha256_file(self.program) != program_sha256:
            raise ValueError("native SIS2 executable identity mismatch")
        self.identity = program_sha256
        self.timeout, self.steps, self.step = timeout, steps, 0
        self.sequence, self.pending = 0, b""
        self.log = (self.case / "native-sis2.log").open("xb")
        self.process = None
        try:
            self.process = subprocess.Popen(
                [str(self.program), str(dt_seconds), str(steps), start, str(elapsed_seconds)],
                cwd=self.case,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
            header = self._until("ZMSIS_READY").split()
            if len(header) != 6 or int(header[1]) != 1 or int(header[-1]) != dt_seconds:
                raise ValueError("native SIS2 protocol/interval mismatch")
            self.shape = tuple(map(int, header[2:5]))
            if any(n <= 0 for n in self.shape):
                raise ValueError("invalid native SIS2 grid dimensions")
        except BaseException:
            self.close(force=True)
            raise

    def _until(self, marker):
        deadline = time.monotonic() + self.timeout
        while True:
            if b"\n" in self.pending:
                line, self.pending = self.pending.split(b"\n", 1)
                text = line.decode("utf8", errors="replace").strip()
                if "Bad ice data" in text or "FATAL" in text:
                    raise RuntimeError("native SIS2 reported invalid input: " + text)
                if text.startswith(marker + " "):
                    return text
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError("native SIS2 reply deadline exceeded")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("native SIS2 exited before " + marker)
            self.log.write(chunk)
            self.log.flush()
            self.pending += chunk

    def exchange(self, ocean, air=None, *, land=None):
        """Publish ocean surface, optionally advance one full fast+slow ice step."""
        advance = air is not None
        if advance and self.step >= self.steps:
            raise ValueError("native ice step limit reached")
        groups = [(ocean, OCEAN_FIELDS, self.shape[:2])]
        if advance:
            if land is None:
                raise ValueError("advancing SIS2 requires explicit land freshwater/enthalpy fields")
            groups += [(air, AIR_FIELDS, self.shape)]
            groups += [(land, LAND_FIELDS, self.shape[:2])]
        elif land is not None:
            raise ValueError("land forcing is only consumed by an ice advance")
        arrays = []
        for values, names, shape in groups:
            if set(values) != set(names):
                raise ValueError("native exchange field set mismatch")
            for name in names:
                value = np.asarray(values[name], dtype="<f8")
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError("invalid native exchange field: " + name)
                arrays.append(value)
        if np.any(ocean["t"] <= 0) or np.any(ocean["s"] < 0) or np.any(ocean["frazil"] < 0):
            raise ValueError("invalid ocean SI temperature, salinity or frazil energy")
        self.sequence += 1
        request = self.case / f"exchange-{self.sequence:08}.bin"
        if "\n" in str(request) or len(str(request)) > 980:
            raise ValueError("native exchange path exceeds protocol limits")
        with request.open("xb") as stream:
            for value in arrays:
                stream.write(value.tobytes(order="F"))
        try:
            self.process.stdin.write(
                (("STEP " if advance else "SURFACE ") + str(request) + "\n").encode()
            )
            self.process.stdin.flush()
            reply = self._until("ZMSIS_REPLY").split()
            expected_step = self.step + int(advance)
            if len(reply) != 2 or int(reply[1]) != expected_step:
                raise ValueError("native ice clock reply mismatch")
            result = decode_reply(str(request) + ".out", self.shape, fluxes=advance)
            self.step = expected_step
            return result
        except BaseException:
            self.close(force=True)
            raise

    def checkpoint(self):
        """Save the full native ice restart, including categories and dynamics.

        This is only the ice half. The coupled runner must commit it together
        with the ocean checkpoint and one common accepted time/identity.
        """
        directory = self.case / "RESTART"
        directory.mkdir(exist_ok=True)
        try:
            self.process.stdin.write(b"SAVE\n")
            self.process.stdin.flush()
            header = self._until("ZMSIS_SAVED").split()
            if len(header) != 2 or int(header[1]) != self.step:
                raise ValueError("native checkpoint clock reply mismatch")
            files = {
                p.relative_to(directory).as_posix(): sha256_file(p)
                for p in sorted(directory.rglob("*"))
                if p.is_file()
            }
            if not any(n.endswith(".nc") for n in files):
                raise ValueError("native SIS2 checkpoint produced no state files")
            return {
                "accepted_ice_steps": self.step,
                "program_sha256": self.identity,
                "native_restart_files": files,
            }
        except BaseException:
            self.close(force=True)
            raise

    def close(self, *, force=False):
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    if force:
                        self.process.kill()
                    else:
                        self.process.stdin.write(b"STOP\n")
                        self.process.stdin.flush()
                    if not force:
                        deadline = time.monotonic() + 5
                        while self.process.poll() is None:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise subprocess.TimeoutExpired(self.process.args, 5)
                            if select.select([self.process.stdout], [], [], min(remaining, 0.1))[0]:
                                tail = os.read(self.process.stdout.fileno(), 65536)
                                self.log.write(tail)
                    self.process.wait(timeout=5)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    self.process.kill()
                    self.process.wait(timeout=5)
            for stream in (self.process.stdin, self.process.stdout):
                stream.close()
        if not self.log.closed:
            self.log.close()
        if not force and sha256_file(self.program) != self.identity:
            raise ValueError("native SIS2 executable changed during integration")
        if not force and self.process is not None and self.process.returncode != 0:
            raise RuntimeError("native SIS2 failed during shutdown")

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close(force=kind is not None)
