"""Native pinned MOM6 wave preparation/execution and strict diagnostic conversion."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

import netCDF4
import numpy as np

from zhenmode.baselines.mom6.adapter import prepare_wave_input
from zhenmode.benchmarks.standing_wave import MOM6, digest, legacy_contract
from zhenmode.evaluation.native_channel import duration
from zhenmode.execution.native_channel import metadata, native_arrays
from zhenmode.execution.runs import write_json
from zhenmode.provenance.sources import load_json, sha256_file


def literal(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value) if isinstance(value, (int, float)) else '"' + value + '"'


def integrate(config_file):
    config = load_json(config_file)
    c, case = config["contract"], Path.cwd()
    # The established input writer keeps its original v0 contract interface.
    # Geometry/initial values are unchanged; the suite declares v1 separately.
    if "benchmark" in c:
        from zhenmode.baselines.mom6.channel_initial import prepare
        prepare(c, case / "INPUT")
    else:
        initial_contract = case / "mom-initial-contract-v0.json"
        grid_case = {64: "coarse", 128: "medium", 256: "fine"}[c["nx"]]
        write_json(initial_contract, legacy_contract(grid_case), create=True)
        prepare_wave_input(initial_contract, case / "INPUT")
    (case / "RESTART").mkdir()
    (case / "MOM_override").write_text("", encoding="utf8")
    settings = dict(
        NIGLOBAL=c["nx"],
        NJGLOBAL=c["ny"],
        NK=c["nz"],
        LAYOUT="1, 1",
        REENTRANT_X=True,
        REENTRANT_Y=False,
        GRID_CONFIG="cartesian",
        AXIS_UNITS="m",
        WESTLON=0.0,
        SOUTHLAT=0.0,
        LENLON=c["Lx_m"],
        LENLAT=c["Ly_m"],
        TOPO_CONFIG="flat",
        MAXIMUM_DEPTH=c["H_m"],
        MINIMUM_DEPTH=0.0,
        ROTATION="beta",
        F_0=c["f"],
        BETA=0.0,
        RHO_0=1025.0,
        G_EARTH=9.81,
        BOUSSINESQ=True,
        EQN_OF_STATE="LINEAR",
        RHO_REF_LINEAR_EOS=1025.0,
        T_REF_LINEAR_EOS=15.0,
        S_REF_LINEAR_EOS=35.0,
        DRHO_DT=-0.205,
        DRHO_DS=0.779,
        DRHO_DP=0.0,
        COORD_CONFIG="none",
        USE_REGRIDDING=False,
        BULKMIXEDLAYER=False,
        THICKNESS_CONFIG="file",
        THICKNESS_FILE="standing_wave_initial.nc",
        INTERFACE_IC_VAR="eta",
        INPUTDIR="INPUT",
        TS_CONFIG="file",
        TS_FILE="standing_wave_initial.nc",
        TEMP_IC_VAR="ptemp",
        SALT_IC_VAR="salt",
        VELOCITY_CONFIG="file" if "benchmark" in c else "zero",
        DT=c["dt"],
        DTBT=c["dt"],
        DT_THERM=c["dt"],
        DT_FORCING=c["dt"],
        SPLIT=True,
        SPLIT_RK2B=False,
        DO_DYNAMICS=True,
        OFFLINE=False,
        ENABLE_THERMODYNAMICS=True,
        ADIABATIC=c.get("benchmark") != "thermal-wind",
        LAPLACIAN=False,
        BIHARMONIC=False,
        KH=0.0,
        AH=0.0,
        KHTR=0.0,
        KHTH=0.0,
        THICKNESSDIFFUSE=False,
        KV=0.0,
        KD=0.0,
        KD_MIN=0.0,
        KD_MAX=0.0,
        KV_MOLECULAR=0.0,
        KV_BBL_MIN=0.0,
        KV_TBL_MIN=0.0,
        KV_EXTRA_BBL=0.0,
        HBBL=10.0,
        HMIX_FIXED=10.0,
        CDRAG=0.0,
        BOTTOMDRAGLAW=False,
        CHANNEL_DRAG=False,
        USE_KPP=False,
        ENERGETICS_SFC_PBL=False,
        USE_CVMix_CONVECTION=False,
        FRAZIL=False,
        DO_GEOTHERMAL=False,
        WIND_CONFIG="zero",
        BUOY_CONFIG="NONE",
        TIMEUNIT=1.0,
        DAYMAX=duration(c),
        SAVE_INITIAL_CONDS=True,
        WRITE_GEOM=1,
        RESTART_CONTROL=0,
    )
    if "benchmark" in c:
        settings.update(VELOCITY_FILE="standing_wave_initial.nc", U_IC_VAR="u", V_IC_VAR="v")
    # LAYOUT is a native integer pair.
    text = (
        "\n".join(
            k + " = " + ("1, 1" if k == "LAYOUT" else literal(v)) for k, v in settings.items()
        )
        + "\n"
    )
    (case / "MOM_input").write_text(text)
    (case / "input.nml").write_text(
        "&MOM_input_nml\n output_directory='./', input_filename='n', restart_input_dir='INPUT/', restart_output_dir='RESTART/', parameter_filename='MOM_input','MOM_override'\n/\n&diag_manager_nml\n/\n&fms_nml\n domains_stack_size=955296, stack_size=0\n/\n"
    )
    (case / "diag_table").write_text(
        f'"Native channel"\n1 1 1 0 0 0\n"prog",{c["output_s"]},"seconds",1,"seconds","Time"\n'
        + "".join(
            '"ocean_model","' + n + '","' + n + '","prog","all",.false.,"none",1\n'
            for n in ["u", "v", "h", "e", "temp", "salt"]
        )
    )
    exe = Path(config["mom_executable"])
    revision = subprocess.check_output(
        [
            "git",
            "-c",
            "safe.directory=" + str(Path(config["mom_source"]).resolve()),
            "-C",
            config["mom_source"],
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()
    if revision != MOM6:
        raise ValueError("MOM6 source differs from fixed wave revision")
    identities = {
        "program_sha256": sha256_file(exe),
        "source_commit": revision,
        "input_files": {
            p.relative_to(case).as_posix(): sha256_file(p)
            for p in case.rglob("*")
            if p.is_file() and p.name != "run.log"
        },
        "build_provenance": "cached binary; full historical source-to-binary build receipt unavailable",
    }
    write_json(case / "native-run.json", identities, create=True)
    subprocess.run([str(exe)], cwd=case, env=dict(os.environ, OMP_NUM_THREADS="1"), check=True)
    if sha256_file(exe) != identities["program_sha256"] or any(
        sha256_file(case / name) != value for name, value in identities["input_files"].items()
    ):
        raise ValueError("MOM6 executable/input changed")


def convert(c, directory, resources):
    case = Path(directory)
    before = load_json(case / "native-run.json")
    text = (case / "run.log").read_text(encoding="utf8")
    settings = (case / "MOM_parameter_doc.all").read_text(encoding="utf8")
    resolved = {}
    for line in settings.splitlines():
        match = re.match(r"^([A-Za-z0-9_]+)\s*=\s*(.*?)\s*!", line)
        if match:
            resolved[match[1]] = match[2].strip().strip('"')
    expected = {
        "SPLIT": "True",
        "SPLIT_RK2B": "False",
        "DT": str(c["dt"]),
        "DT_THERM": str(c["dt"]),
        "DT_FORCING": str(c["dt"]),
        "DTBT": str(c["dt"]),
        "DO_DYNAMICS": "True",
        "OFFLINE_TRACER_MODE": "False",
        "USE_REGRIDDING": "False",
        "RHO_0": "1025.0",
        "G_EARTH": "9.81",
        "F_0": str(c["f"]),
        "BETA": "0.0",
        "KV": "0.0",
        "KHTR": "0.0",
        "LAPLACIAN": "False",
        "BIHARMONIC": "False",
        "BOTTOMDRAGLAW": "False",
        "THICKNESSDIFFUSE": "False",
        "BULKMIXEDLAYER": "False",
        "ADIABATIC": "False" if c.get("benchmark") == "thermal-wind" else "True",
    }
    for k, v in expected.items():
        actual = resolved.get(k)
        match = actual == v
        if k == "F_0" and c["f"] != 0 and actual is not None:
            match = np.isclose(float(actual), c["f"], rtol=5e-12, atol=0)
        if not match:
            raise ValueError(f"MOM6 resolved {k}={resolved.get(k)!r}, expected {v!r}")
    write_json(
        case / "resolved-options.json",
        dict(checked_options=expected, actual_options=resolved),
        create=True,
    )
    nx, ny = c["nx"], c["ny"]
    dx, dy = c["Lx_m"] / nx, c["Ly_m"] / ny
    with (
        netCDF4.Dataset(case / "prog.nc") as d,
        netCDF4.Dataset(case / "MOM_IC.nc") as ic,
        netCDF4.Dataset(case / "ocean_geometry.nc") as geom,
    ):
        if not np.array_equal(d["Time"][:], np.arange(c["output_s"], duration(c) + 1, c["output_s"])):
            raise ValueError("MOM6 native output times are incomplete or changed")
        if not np.all(geom["wet"][:] == 1):
            raise ValueError("MOM6 native wet mask differs from the all-wet case")
        np.testing.assert_allclose(geom["D"][:], c["H_m"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(geom["Ah"][:], dx * dy, rtol=1e-13)

        def field(name, initial):
            value = d[name][:]
            if name == "v":
                mask = np.ma.getmaskarray(value)
                if mask[:, :, 1:-1].any():
                    raise ValueError("unknown missing interior normal velocities")
                # Native FMS does not write closed-wall velocity diagnostics. Physical BC is v=0.
                if not (np.all(mask[:, :, 0]) and np.all(mask[:, :, -1])):
                    raise ValueError("MOM6 wall diagnostics do not match the declared boundary masks")
                value = value.filled(0.0)
            else:
                if np.ma.getmaskarray(value).any():
                    raise ValueError("masked interior prognostic values")
            return np.concatenate(
                (np.asarray(ic[initial][:]), np.asarray(value)), axis=0
            ).transpose(0, 3, 2, 1)

        h = field("h", "h")
        T = field("temp", "Temp")
        S = field("salt", "Salt")
        u = field("u", "u")
        v = field("v", "v")
        np.testing.assert_allclose(u[:, -1], u[:, 0], rtol=0, atol=1e-14)
        u = u[:, :-1]
        eta = np.concatenate(
            (np.asarray(ic["eta"][:, 0]), np.asarray(d["e"][:, 0])), axis=0
        ).transpose(0, 2, 1)
        ux, uy = np.meshgrid(np.asarray(d["xq"][:-1]), np.asarray(d["yh"][:]), indexing="ij")
        vx, vy = np.meshgrid(np.asarray(d["xh"][:]), np.asarray(d["yq"][:]), indexing="ij")
        mx, my = np.meshgrid(np.asarray(d["xh"][:]), np.asarray(d["yh"][:]), indexing="ij")
    initialization_s = float(re.search(r"^Initialization\s+1\s+(\S+)", text, re.M)[1])
    integration_s = float(re.search(r"^Main loop\s+1\s+(\S+)", text, re.M)[1])
    fields = dict(
        time=np.arange(0, duration(c) + 1, c["output_s"], dtype=float),
        eta=eta,
        h=h,
        T=T,
        S=S,
        u=u,
        v=v,
    )
    information = metadata(
        c,
        "MOM6",
        source_sha=before["source_commit"],
        executable_sha256=before["program_sha256"],
        input_sha256=before["input_files"]["INPUT/standing_wave_initial.nc"],
        config_sha256=digest(before["input_files"]),
        initialization_s=initialization_s,
        integration_s=integration_s,
        resources=resources,
        numerics=dict(
            time_scheme="native SPLIT=True SPLIT_RK2B=False",
            transport="full momentum and T/S transport",
            filters="native numerical filtering; no added sponge",
            vertical_coordinate=f"{c['nz']} moving native layer interfaces, no ALE remapping",
            substeps="DT=DTBT=DT_THERM=DT_FORCING",
            resolved_options=c["mom_time_options"],
        ),
    )
    return native_arrays(
        c, fields, layout="cgrid", x_eta=mx, y_eta=my, x_u=ux, y_u=uy, x_v=vx, y_v=vy
    ), information


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-config", required=True)
    integrate(parser.parse_args().worker_config)
