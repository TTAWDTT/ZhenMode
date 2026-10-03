"""Regression tests for the polar-cap blend orientation.

`_polar_cap_weights` returns the blend weights in POLE-INWARD order: weight 1
on the pole row, cos^2 taper into the interior. `_apply_polar_cap` slices the
south band inward from its own pole, so that order matches. The north band
slices POLE-LAST (`field[:, -nb:]` starts nb-1 rows INSIDE the wall), so its
weights have to be reversed -- and were not.

The consequence is invisible until a deep-water wall meets it: the north cap
spends its full zonal mean a row band inside the wall and leaves the wall row
itself at 0.15 of it, i.e. effectively uncapped. A 2dx zonal checkerboard then
grows there and doubles every step. Measured on the real ETOPO2022 relief at
1 deg / ny=120: peak |u| pinned to j=ny-1, 1.0 -> 24.8 m/s between day 1.0 and
day 2.0, while the correctly-capped south wall stayed flat.

Nothing in the suite pinned this mapping: every solver test builds its grid
with polar_cap_rows=0.

Invariants under test:

1. Both pole bands get the FULL zonal average, so the pole rows come out
   zonally uniform: `test_polar_cap_is_pole_anchored_on_both_bands`.
2. The cap commutes with a north-south flip -- the north band is the south
   band seen through a mirror and nothing else:
   `test_polar_cap_is_mirror_symmetric`.
3. The blend weight falls off monotonically from the pole into the interior,
   so the cap edge is never a cliff:
   `test_polar_cap_taper_is_monotone_inward`.

Run:  python -m pytest tests/test_polar_cap.py -v
"""
import os
from types import SimpleNamespace

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from ocean_solver.numerics.horizontal import _apply_polar_cap
from tests.support.grid import all_wet_grid

NCAP = 2
NTAPER = 3
NB = NCAP + NTAPER


def _params(ncap=NCAP, ntaper=NTAPER):
    """_apply_polar_cap reads only these two fields off the params tuple."""
    return SimpleNamespace(polar_cap_rows=ncap, polar_cap_taper=ntaper)


def _checkerboard(nx, ny, amp=1.0):
    """+/-amp alternating in x: the 2dx zonal mode the cap exists to kill."""
    col = np.where(np.arange(nx) % 2 == 0, amp, -amp)[:, None]
    return np.broadcast_to(col, (nx, ny)).copy()


def test_polar_cap_is_pole_anchored_on_both_bands():
    grid = all_wet_grid()
    nx, ny = grid.nx, grid.ny
    field = _checkerboard(nx, ny)
    out = np.asarray(_apply_polar_cap(jnp.asarray(field),
                                      jnp.asarray(grid.wet_mask), _params()))

    # nx is even, so the zonal mean of the checkerboard is exactly 0: a capped
    # row must come out flat at 0, an uncapped one stays at +/-1.
    for name, rows in (("south", range(NCAP)), ("north", range(ny - NCAP, ny))):
        for j in rows:
            row = out[:, j]
            assert np.allclose(row, 0.0), (
                f"{name} pole row j={j} is not zonally uniform after the cap "
                f"(still |max|={np.abs(row).max():.3f}): the blend weights are "
                "anchored at the wrong end of the band")

    # Rows outside the band are untouched.
    assert np.array_equal(out[:, NB:ny - NB], field[:, NB:ny - NB])


def test_polar_cap_is_mirror_symmetric():
    """cap(flip(f)) == flip(cap(f)): the north band is the south band mirrored."""
    grid = all_wet_grid()
    nx, ny = grid.nx, grid.ny
    rng = np.random.default_rng(0)
    field = rng.standard_normal((nx, ny))          # nothing symmetric in it
    wm = jnp.asarray(grid.wet_mask)

    out = np.asarray(_apply_polar_cap(jnp.asarray(field), wm, _params()))
    flipped = np.asarray(_apply_polar_cap(jnp.asarray(field[:, ::-1]), wm,
                                          _params()))
    assert np.allclose(out, flipped[:, ::-1]), (
        "the north cap is not the south cap mirrored: its blend weights are "
        "applied in the wrong row order")


def test_polar_cap_taper_is_monotone_inward():
    grid = all_wet_grid()
    nx, ny = grid.nx, grid.ny
    out = np.asarray(_apply_polar_cap(jnp.asarray(_checkerboard(nx, ny)),
                                      jnp.asarray(grid.wet_mask), _params()))

    # Field amplitude is 1, so the residual amplitude 1 - weight is what the
    # blend leaves behind. One column of i is a meridional profile.
    amp = np.abs(out[0, :])
    assert amp[0] == 0.0, "the pole row must take the full zonal mean"
    assert np.all(np.diff(amp[:NB]) >= -1e-12), (
        "the taper re-sharpens going inward, which is the cliff it exists to "
        "avoid")
    assert np.allclose(amp[NB], 1.0), "the cap must stop at the band edge"
