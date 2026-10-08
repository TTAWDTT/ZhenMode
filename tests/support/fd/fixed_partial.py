"""Shared stepped-bed fixture with independently specified node capacities."""
from dataclasses import replace

import numpy as np

from tests.support.grid import all_wet_grid
from zhenmode.model.config import PhysicsConfig
from zhenmode.model.solver.factory import make_solver_global


def fixed_partial_case(cross_nodes=False, thermodynamics="linear", **options):
    grid = all_wet_grid(nx=4, ny=4, nz=4)
    # One repeat per latitude. Widths are independently specified, including
    # last-node extension beyond a global midpoint and one-node shallow water.
    bed = np.tile(np.array([0.5, 7.0, 27.0, 0.0])[:, None], (1, 4))
    widths = np.tile(
        np.array(
            [
                [0.5, 0.0, 0.0, 0.0],
                [2.5, 4.5, 0.0, 0.0],
                [2.5, 7.5, 17.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )[:, None, :],
        (1, 4, 1),
    )
    if cross_nodes:
        bed = np.tile(np.array([3.0, 14.0, 27.0, 30.0])[:, None], (1, 4))
        widths = np.tile(
            np.array(
                [
                    [3.0, 0.0, 0.0, 0.0],
                    [2.5, 11.5, 0.0, 0.0],
                    [2.5, 7.5, 17.0, 0.0],
                    [2.5, 7.5, 12.5, 7.5],
                ]
            )[:, None, :],
            (1, 4, 1),
        )
        bed[2, 2] = 30.0
        widths[2, 2] = [2.5, 7.5, 12.5, 7.5]
        bed[3, 3] = 0.0
        widths[3, 3] = 0.0
    grid = replace(
        grid,
        z=np.array([0.0, -5.0, -15.0, -30.0]),
        dz=np.array([5.0, 10.0, 15.0]),
        depth=bed,
        wet_mask_3d=(widths > 0).astype(float),
        wet_mask=(bed > 0).astype(float),
        ocean_mask=bed > 0,
        land_mask=bed == 0,
        f=np.zeros((4, 4)),
    )
    physics = replace(
        PhysicsConfig(),
        nu_h=0.0,
        nu_v=0.0,
        nu_bi=0.0,
        kappa_h=0.0,
        kappa_v=0.0,
        kappa_bi=0.0,
        kappa_conv=0.0,
        kappa_gm=0.0,
        kappa_redi=0.0,
        r_bot=0.0,
    )
    settings = dict(
        column_geometry="fixed_partial_v1",
        conservative_kv=True,
        localize_conv=True,
        mode_split=True,
        dt_bt=0.5,
        polar_cap_rows=0,
        polar_cap_taper=0,
        return_params=True,
        use_scan=True,
    )
    settings.update(options)
    if thermodynamics == "teos10_reference":
        physics = replace(physics, thermodynamics=thermodynamics)
        settings["eos_pressure_dbar"] = np.broadcast_to(
            -grid.z * 1025.0 * 9.81 / 10000.0, widths.shape
        ).copy()
    return grid, widths, make_solver_global(grid, physics, 1.0, **settings)

