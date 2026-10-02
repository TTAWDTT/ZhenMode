"""Independent synthetic column fixture shared by bridge and migration tests."""
import numpy as np


def fixture(eta=-2.49, profile="linear"):
    d = np.array([0.0, 5, 15, 30, 100, 300, 500, 1000])
    w = np.array([2.5, 7.5, 12.5, 42.5, 135, 200, 350, 750])
    mask = np.array([1, 1, 1, 1, 1, 1, 1, 0])
    x = -d
    temp = 20 + 0.005 * x
    if profile == "uniform":
        temp[:] = 15
    if profile == "curved":
        temp = 20 + 0.005 * x + 1e-6 * x * x
    v = np.array([temp, np.full(8, 35), 0.001 * x, -0.0005 * x]).T
    return dict(
        depth=d,
        reference_weights=w,
        wet_mask=mask,
        values=v,
        eta=eta,
        Tref=15.0,
        Sref=35.0,
        alpha=2e-4,
        beta=7.6e-4,
        rho0=1025.0,
        gravity=9.81,
        terrain_depth=750.0,
        discrete_bottom=750.0,
        control_interfaces=np.array([]),
        source_sha=np.array("2" * 40),
        terrain_sha=np.array("a" * 64),
    )
