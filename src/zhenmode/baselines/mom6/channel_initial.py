"""Prepare native MOM6 layer/scalar/vector inputs for the frozen channel cases."""

from pathlib import Path

import netCDF4
import numpy as np

from zhenmode.benchmarks.channel_dynamics import initial_native, validate_contract
from zhenmode.benchmarks.standing_wave import digest
from zhenmode.execution.runs import write_json
from zhenmode.provenance.sources import sha256_file


def prepare(c, directory):
    validate_contract(c)
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    values = initial_native(c, "MOM6")
    nx, ny, nz = c["nx"], c["ny"], c["nz"]
    eta, h = values["eta"], values["h"]
    interfaces = np.concatenate((eta[..., None], eta[..., None] - np.cumsum(h, axis=-1)), axis=-1)
    path = directory / "standing_wave_initial.nc"
    with netCDF4.Dataset(path, "w") as ds:
        for name, size in (
            ("x", nx),
            ("y", ny),
            ("xq", nx + 1),
            ("yq", ny + 1),
            ("Layer", nz),
            ("Interface", nz + 1),
        ):
            ds.createDimension(name, size)
        for name, array, axis in (
            ("x", (np.arange(nx) + 0.5) * c["Lx_m"] / nx, "X"),
            ("y", (np.arange(ny) + 0.5) * c["Ly_m"] / ny, "Y"),
            ("xq", np.arange(nx + 1) * c["Lx_m"] / nx, "X"),
            ("yq", np.arange(ny + 1) * c["Ly_m"] / ny, "Y"),
        ):
            var = ds.createVariable(name, "f8", (name,))
            var[:] = array
            var.units = "m"
            var.cartesian_axis = axis
        var = ds.createVariable("eta", "f8", ("Interface", "y", "x"))
        var[:] = interfaces.transpose(2, 1, 0)
        var.units = "m"
        var.positive = "up"
        for name, key, unit in (("ptemp", "T", "degC"), ("salt", "S", "psu")):
            var = ds.createVariable(name, "f8", ("Layer", "y", "x"))
            var[:] = values[key].transpose(2, 1, 0)
            var.units = unit
        u = np.concatenate((values["u"], values["u"][:1]), axis=0)
        var = ds.createVariable("u", "f8", ("Layer", "y", "xq"))
        var[:] = u.transpose(2, 1, 0)
        var.units = "m/s"
        var = ds.createVariable("v", "f8", ("Layer", "yq", "x"))
        var[:] = np.zeros((nz, ny + 1, nx))
        var.units = "m/s"
        ds.contract_sha256 = digest(c)
    write_json(
        directory / "preparation.json",
        dict(
            contract_sha256=digest(c),
            input_sha256=sha256_file(path),
            native_vertical_representation="layer means; fixed reference interfaces and displaced top",
            execution_status="proposed",
        ),
        create=True,
    )
    return path
