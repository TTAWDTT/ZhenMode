"""Regression tests for the vertical-diffusion seafloor boundary condition.

`_d2_dz2` is the node-form vertical diffusion: the default
(`conservative_kv=False`) path for `kappa_v`, and the operator the momentum
vertical diffusion always uses. Its centered stencil spans k-1..k+1, so at a
column whose seafloor is NOT the last grid level the bottom WET node used to
read the ghost layers -- which hold the `T_ref`/`S_ref` sentinel from
`init_state`, not the bottom value. That is a spurious seafloor flux of up to
0.05 K/day: it warms every cold shelf toward +15 C and cools every warm one.
D8 fixed the same defect for the closure operators; D20 in docs/decisions.md
fixes it here.

Invariants under test:

1. The tendency at every WET node is independent of what the ghost layers
   hold. Regression: `test_wet_tendency_ignores_the_ghost_sentinel`.
2. The bottom wet node sees a ONE-SIDED (no-flux) stencil, not a centered one
   reaching through the floor: `test_bottom_wet_node_is_one_sided`.
3. A column that IS wet to the last grid level is untouched -- the fill is a
   no-op there, so the legacy centered stencil survives bit-for-bit:
   `test_deep_column_keeps_the_centered_stencil`.

Run:  python -m pytest tests/test_vertical_bc.py -v
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from zhenmode.model.config.definitions import PhysicsConfig
from zhenmode.model.geometry.fd import make_fd_params
from zhenmode.model.geometry.types import GlobalOceanGrid
from zhenmode.model.numerics.vertical import _d2_dz2

T_REF = PhysicsConfig().T_ref       # 15.0
SENTINEL = 999.0                    # deliberately absurd, so a leak is loud

NZ = 6
Z = -np.linspace(50.0, 4000.0, NZ)  # -50 .. -4000
SHALLOW = 1500.0                    # wet only at k=0,1 -> bottom wet node k=1


def _grid(depth_shallow=SHALLOW, nx=8, ny=6):
    """Global grid with a shallow half (ghost layers) and a deep half."""
    lat = np.linspace(-40.0, 40.0, ny)
    lon = np.linspace(0.5, 359.5, nx)
    R = 6.371e6
    cos_lat = np.cos(np.radians(lat))
    depth = np.full((nx, ny), 4000.0)
    depth[: nx // 2, :] = depth_shallow
    wet = np.ones((nx, ny))
    wet3 = (np.abs(Z)[None, None, :] <= depth[:, :, None]).astype(float)
    return GlobalOceanGrid(
        lon=lon, lat=lat,
        dx_2d=np.broadcast_to(R * cos_lat * np.radians(360.0 / nx),
                              (nx, ny)).copy(),
        dy=float(R * np.radians(abs(lat[1] - lat[0]))), cos_lat=cos_lat,
        f=np.broadcast_to(2 * 7.2921e-5 * np.sin(np.radians(lat)),
                          (nx, ny)).copy(),
        z=Z, dz=-np.diff(Z), nz=NZ, depth=depth, wet_mask=wet,
        ocean_mask=np.ones((nx, ny), dtype=bool), land_mask=np.zeros((nx, ny)),
        wet_mask_3d=wet3, nx=nx, ny=ny)


def _profile(grid, ghosts):
    """Warm shallow water over cold deep water; `ghosts` fills the dry layers."""
    T = np.full(grid.wet_mask_3d.shape, ghosts)
    for k in range(NZ):
        wet_k = grid.wet_mask_3d[:, :, k] > 0.5
        T[:, :, k] = np.where(wet_k, 20.0 - 2.0 * k, T[:, :, k])
    return T


def _kbot(grid):
    idx = (grid.wet_mask_3d > 0.5) * np.arange(NZ)
    return idx.max(axis=-1).astype(int)


def test_wet_tendency_ignores_the_ghost_sentinel():
    """A ghost-layer value must never reach a wet node's tendency."""
    grid = _grid()
    p = make_fd_params(grid)
    wet = grid.wet_mask_3d > 0.5

    sane = np.array(_d2_dz2(jnp.array(_profile(grid, T_REF)), p))
    junk = np.array(_d2_dz2(jnp.array(_profile(grid, SENTINEL)), p))

    assert np.allclose(sane[wet], junk[wet], rtol=0.0, atol=0.0), (
        "the ghost layers leaked into the wet column")


def test_bottom_wet_node_is_one_sided():
    """At the seafloor the stencil must be one-sided, not centered."""
    grid = _grid()
    p = make_fd_params(grid)
    T = _profile(grid, T_REF)
    got = np.array(_d2_dz2(jnp.array(T), p))

    kbot = _kbot(grid)
    hm = np.asarray(p.d2z_hm)[0, 0]          # |z[k-1] - z[k]|
    hp = np.asarray(p.d2z_hp)[0, 0]          # |z[k+1] - z[k]|
    denom = np.asarray(p.d2z_denom)[0, 0]
    k = int(kbot[0, 0])                      # shallow half -> k = 1
    assert k == 1, "expected the shallow column to be two levels deep"

    # The centered stencil with the ghost replaced by the bottom value:
    #   (T[k+1]*hm + T[k-1]*hp - T[k]*(hm+hp)) / denom, with T[k+1] := T[k]
    expected = hp[k - 1] * (T[0, 0, k - 1] - T[0, 0, k]) / denom[k - 1]
    assert np.isclose(got[0, 0, k], expected, rtol=1e-13, atol=0.0)

    # ... and that is NOT what a centered stencil through the floor gives.
    centered = ((T[0, 0, k + 1] * hm[k - 1] + T[0, 0, k - 1] * hp[k - 1]
                 - T[0, 0, k] * (hm[k - 1] + hp[k - 1])) / denom[k - 1])
    assert not np.isclose(got[0, 0, k], centered, rtol=0.0, atol=0.0)


def test_deep_column_keeps_the_centered_stencil():
    """A column wet to the last level has no ghost layer: nothing changes."""
    grid = _grid()
    p = make_fd_params(grid)
    T = _profile(grid, T_REF)
    got = np.array(_d2_dz2(jnp.array(T), p))

    hm = np.asarray(p.d2z_hm)[0, 0]
    hp = np.asarray(p.d2z_hp)[0, 0]
    denom = np.asarray(p.d2z_denom)[0, 0]
    i = grid.nx - 1                          # deep half -> kbot = nz-1
    for k in range(1, NZ - 1):
        expected = ((T[i, 0, k + 1] * hm[k - 1] + T[i, 0, k - 1] * hp[k - 1]
                     - T[i, 0, k] * (hm[k - 1] + hp[k - 1])) / denom[k - 1])
        assert np.isclose(got[i, 0, k], expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    test_wet_tendency_ignores_the_ghost_sentinel()
    test_bottom_wet_node_is_one_sided()
    test_deep_column_keeps_the_centered_stencil()
    print("ALL PASS")
