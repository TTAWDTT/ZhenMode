"""Controlled production CLI fixture shared by restart and monitoring tests."""
import sys
from dataclasses import replace

import numpy as np
import pytest

import run_long_integration_global as driver
from tests.support.grid import all_wet_grid


def run_controlled_driver(monkeypatch, directory, *, restart=None, crash_after=None,
                          save_3d=False, options=(), step_override=None, init_override=None, expected_code=0):
    grid = replace(all_wet_grid(nx=8, ny=8, nz=4), f=np.zeros((8, 8)))
    monkeypatch.setattr(driver, "make_global_grid", lambda *args, **kwargs: grid)
    monkeypatch.setattr(driver, "get_initial_fields", lambda grid:
                        (np.full((8, 8, 4), 17.), np.full((8, 8, 4), 35.)))
    monkeypatch.setattr(driver, "build_seasonal_wind_global", lambda grid, year:
                        [(np.full((8, 8), (month + 1) * 0.001), np.zeros((8, 8)))
                         for month in range(12)])
    original_factory = driver.make_solver_global
    contracts = []
    original_contract = driver.make_restart_contract

    def build_contract(*args, **kwargs):
        contract = original_contract(*args, **kwargs)
        contracts.append(contract)
        return contract

    monkeypatch.setattr(driver, "make_restart_contract", build_contract)

    def factory(*args, **kwargs):
        result = list(original_factory(*args, **kwargs))
        original_step = result[-1]
        if init_override is not None:
            initialize = result[1]
            result[1] = lambda **fields: init_override(initialize(**fields))
        calls = 0

        def step(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == crash_after:
                raise RuntimeError("controlled interruption")
            if step_override is not None:
                return step_override(args[0], calls)
            return original_step(*args, **kwargs)

        result[-1] = step
        return tuple(result)

    monkeypatch.setattr(driver, "make_solver_global", factory)
    arguments = ["ocean-solver", "--days", str(80. / 86400.), "--dt", "10", "--dt-bt", "5",
                 "--mode-split", "--seasonal-wind", "--snap-days", str(20. / 86400.),
                 "--checkpoint-days", str(20. / 86400.), "--tag", "controlled",
                 "--out-dir", str(directory), "--log-dir", str(directory), "--nu-h", "0",
                 "--nu-bi", "0", "--kappa-v", "0", "--kappa-conv", "0",
                 "--polar-cap-rows", "0", "--polar-cap-taper", "0",
                 "--no-meridional-heat-flux", "--no-bulk-flux"]
    if save_3d:
        arguments.extend(["--save-3d", "--save-3d-terms"])
    if restart:
        arguments.extend(["--restart-from", str(restart)])
    arguments.extend(options)
    monkeypatch.setattr(sys, "argv", arguments)
    previous_stdout = sys.stdout
    try:
        if crash_after is None:
            assert driver.main() == expected_code
        else:
            with pytest.raises(RuntimeError, match="controlled interruption"):
                driver.main()
    finally:
        if isinstance(sys.stdout, driver._Tee):
            sys.stdout.file.close()
        sys.stdout = previous_stdout
    return grid, contracts[0] if contracts else None
