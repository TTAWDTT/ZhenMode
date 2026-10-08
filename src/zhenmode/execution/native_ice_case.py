"""FD-compatible native SIS2 geometry and explicit coupling configuration.

The supergrid carries exact spherical rectangle areas. The native analytic
spherical grid uses midpoint areas and is not interchangeable in stock checks.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np

from zhenmode.provenance.sources import sha256_file


def prepare_sis2_case(native_prepared, output, *, start="1958-01-01T00:00:00"):
    """Prepare a shared rectangular grid, preserving approved bed and wet cells.

    Native SIS2 retains its physical defaults, except shared radius, density,
    gravity, heat-capacity convention and A-grid stress exchange. This is a
    coupling candidate; radiation, full physics and long-run qualification
    remain properties of the eventual run, not this geometry preparation.
    """
    from zhenmode.execution.native_fd import load_fd_native_inputs
    from zhenmode.model.config import CP0_TEOS10, G_EARTH, R_EARTH, RHO_0

    timestamp = datetime.fromisoformat(start)
    if timestamp.tzinfo is not None or timestamp.isoformat(timespec="seconds") != start:
        raise ValueError("case start must be Gregorian YYYY-MM-DDTHH:MM:SS")
    native_prepared, output = Path(native_prepared).resolve(), Path(output).resolve()
    grid, _, _, _ = load_fd_native_inputs(native_prepared)
    if grid.nx < 3 or grid.ny < 3:
        raise ValueError("native SIS2 grid needs at least three cells per axis")
    dlon, dlat = 360 / grid.nx, 180 / grid.ny
    west, south = grid.lon[0] - dlon / 2, grid.lat[0] - dlat / 2
    if (
        not np.allclose(grid.lon, west + dlon * (np.arange(grid.nx) + 0.5), atol=1e-10, rtol=0)
        or not np.allclose(grid.lat, -90 + dlat * (np.arange(grid.ny) + 0.5), atol=1e-10, rtol=0)
        or not np.isclose(south, -90, atol=1e-10, rtol=0)
    ):
        raise ValueError("native SIS2 case requires regular global rectangular cell centres")
    source_id = sha256_file(native_prepared / "fd-native-inputs.npz")
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "INPUT"
    inputs.mkdir()
    lon = np.linspace(west, west + 360, 2 * grid.nx + 1)
    lat = np.linspace(-90, 90, 2 * grid.ny + 1)
    x, y = np.meshgrid(lon, lat)
    dx = np.broadcast_to(
        (R_EARTH * np.deg2rad(dlon / 2) * np.cos(np.deg2rad(lat)))[:, None],
        (2 * grid.ny + 1, 2 * grid.nx),
    ).copy()
    dy = np.full((2 * grid.ny, 2 * grid.nx + 1), R_EARTH * np.deg2rad(dlat / 2))
    area = np.broadcast_to(
        (R_EARTH**2 * np.deg2rad(dlon / 2) * np.diff(np.sin(np.deg2rad(lat))))[:, None],
        (2 * grid.ny, 2 * grid.nx),
    ).copy()
    coarse_area = area.reshape(grid.ny, 2, grid.nx, 2).sum(axis=(1, 3)).T
    if not np.allclose(coarse_area, grid.dx_2d * grid.dy, atol=1e-6, rtol=1e-10):
        raise ValueError("SIS2 supergrid cell areas disagree with FD reference geometry")
    with netCDF4.Dataset(inputs / "ocean_hgrid.nc", "w") as data:
        for name, size in (
            ("nx", 2 * grid.nx),
            ("ny", 2 * grid.ny),
            ("nxp", 2 * grid.nx + 1),
            ("nyp", 2 * grid.ny + 1),
            ("string", 255),
        ):
            data.createDimension(name, size)
        for name, values, dimensions, unit in (
            ("x", x, ("nyp", "nxp"), "degrees_east"),
            ("y", y, ("nyp", "nxp"), "degrees_north"),
            ("dx", dx, ("nyp", "nx"), "m"),
            ("dy", dy, ("ny", "nxp"), "m"),
            ("area", area, ("ny", "nx"), "m2"),
            ("angle_dx", np.zeros_like(x), ("nyp", "nxp"), "degrees"),
        ):
            variable = data.createVariable(name, "f8", dimensions)
            variable.units = unit
            variable[:] = values
        data.createVariable("tile", "S1", ("string",))[:5] = np.array(list("tile1"), dtype="S1")
    with netCDF4.Dataset(inputs / "topog.nc", "w") as data:
        for name, values, unit, axis in (
            ("xh", grid.lon, "degrees_east", "X"),
            ("yh", grid.lat, "degrees_north", "Y"),
        ):
            data.createDimension(name, len(values))
            coordinate = data.createVariable(name, "f8", (name,))
            coordinate.units, coordinate.axis = unit, axis
            coordinate[:] = values
        depth = data.createVariable("depth", "f8", ("yh", "xh"))
        depth.units = "m"
        depth[:] = np.where(grid.ocean_mask, grid.depth, 0).T
    (output / "SIS_input").write_text(
        f"NIGLOBAL = {grid.nx}\nNJGLOBAL = {grid.ny}\nLAYOUT = 1, 1\n"
        'GRID_CONFIG = "mosaic"\nGRID_FILE = "ocean_hgrid.nc"\n'
        'TOPO_CONFIG = "file"\nTOPO_FILE = "topog.nc"\nINPUTDIR = "INPUT"\n'
        "MINIMUM_DEPTH = 0.\nTRIPOLAR_N = False\nCGRID_ICE_DYNAMICS = True\n"
        'ICE_OCEAN_STRESS_STAGGER = "A"\n'
        f"RAD_EARTH = {R_EARTH:.17g}\nRHO_OCEAN = {RHO_0:.17g}\nG_EARTH = {G_EARTH:.17g}\n"
        f"CP_SEAWATER = {CP0_TEOS10:.17g}\nCP_BRINE = {CP0_TEOS10:.17g}\nVERBOSITY = 2\n",
        encoding="utf8",
    )
    (output / "input.nml").write_text(
        "&SIS_input_nml\n parameter_filename='SIS_input', output_directory='.',\n"
        " restart_input_dir='INPUT', restart_output_dir='RESTART', input_filename='n'\n/\n"
        "&fms_nml\n clock_grain='component'\n/\n&diag_manager_nml\n/\n"
        "&sat_vapor_pres_nml\n construct_table_wrt_liq=.true., construct_table_wrt_liq_and_ice=.true.\n/\n"
        "&surface_flux_nml\n ncar_ocean_flux=.true., ncar_ocean_flux_orig=.false.,\n"
        " raoult_sat_vap=.true., alt_gustiness=.true., gust_const=.5,\n"
        " use_virtual_temp=.true., no_neg_q=.false., use_mixing_ratio=.false., do_simple=.false.\n/\n",
        encoding="utf8",
    )
    (output / "diag_table").write_text(
        '"FD-native SIS2 coupling candidate"\n' + timestamp.strftime("%Y %m %d %H %M %S") + "\n",
        encoding="utf8",
    )
    files = {
        p.relative_to(output).as_posix(): sha256_file(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    receipt = {
        "status": "prepared_native_SIS2_geometry_candidate",
        "start": start,
        "shape": [grid.nx, grid.ny],
        "wet_cells": int(grid.ocean_mask.sum()),
        "native_arrays_sha256": source_id,
        "native_receipt_sha256": sha256_file(native_prepared / "fd_initialization.json"),
        "files": files,
        "native_read_verified": False,
        "full_case_qualification": False,
        "coupling_scope": "same rectangular positions, bed, wet support and exact spherical areas",
        "physics_scope": "native SIS2 defaults with listed shared constants; full-case choices unqualified",
    }
    if sha256_file(native_prepared / "fd-native-inputs.npz") != source_id:
        raise ValueError("FD native input changed while preparing the sea-ice case")
    (output / "case.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf8")
    return receipt
