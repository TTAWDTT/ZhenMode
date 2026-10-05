"""Production mixed-layer depth configuration contract."""
import numpy as np


def test_solver_accepts_mixed_layer_depth():
    from dataclasses import replace

    from tests.support.grid import all_wet_grid
    from zhenmode.model.config import PhysicsConfig
    from zhenmode.model.solver.factory import make_solver_global

    grid = all_wet_grid(nx=16, ny=8, nz=4)
    physics = replace(PhysicsConfig(), nu_h=0.0, nu_bi=0.0,
                      nu_v=0.0, kappa_h=0.0, kappa_v=0.0,
                      kappa_conv=0.0, kappa_gm=0.0, kappa_redi=0.0)
    forcing = tuple(np.zeros((grid.nx, grid.ny)) for _ in range(3))
    _, _, _, params, _ = make_solver_global(
        grid, physics, 60.0, forcing=forcing, lambda_bulk=0.0,
        mixed_layer_depth_m=50.0, T_init=None, S_init=None,
        polar_cap_rows=0, polar_cap_taper=0,
        mode_split=False, dtype='float64', return_params=True)
    assert params.mixed_layer_depth_m == 50.0
