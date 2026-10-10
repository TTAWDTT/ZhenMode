"""Independent native initializer; Julia consumes these actual initial arrays."""

from pathlib import Path

import netCDF4

from zhenmode.benchmarks.channel_dynamics import initial_native, validate_contract
from zhenmode.benchmarks.standing_wave import digest
from zhenmode.provenance.sources import sha256_file


def prepare(c, directory):
    validate_contract(c)
    values = initial_native(c, "Oceananigans")
    path = Path(directory) / "initial-oceananigans.nc"
    if path.exists():
        raise FileExistsError(path)
    with netCDF4.Dataset(path, "w") as ds:
        for name, size in (("x", c["nx"]), ("y", c["ny"]), ("z", c["nz"])):
            ds.createDimension(name, size)
        # Python top-to-bottom becomes Julia bottom-to-top and x/y/z order.
        ds.createVariable("eta", "f8", ("y", "x"))[:] = values["eta"].T
        for name in ("T", "S", "u"):
            ds.createVariable(name, "f8", ("z", "y", "x"))[:] = values[name][..., ::-1].transpose(
                2, 1, 0
            )
        ds.contract_sha256 = digest(c)
    return path, sha256_file(path)
