"""Regression tests for the nu_h subcycle sizing in the split L half-steps.

`nu_nsub='cfl'` used to size from the meridional spacing alone:

    n = ceil(nu_h * dt / (0.25 * dy**2))

with `dy` the scalar 1-deg spacing (111.2 km). The 5-point Laplacian's most
negative eigenvalue sums BOTH metric terms, and the zonal spacing is the
smaller one: `dx = dy*cos(lat)`, so `1/dx^2` is 3.9x `1/dy^2` at the poleward
rows. The old formula returned 6 where the bound needs 15 -- an LHS of 0.592
against the 0.5 FTCS bound, i.e. straight through it.

That never surfaced in production, for two reasons: every long run left
`--nu-nsub` at its default (`n_nu = n_subcyc` = 24, safely inside the bound),
and the polar cap zonally averages exactly the rows that would diverge -- with
the cap on, n=6 and n=24 are indistinguishable over 4 days on the real relief
(see docs/decisions.md D12 and D21).

Invariants under test:

1. The production 1-deg geometry sizes to 15 and satisfies the FTCS bound at
   the true worst point, minimally: `test_cfl_count_satisfies_the_2d_bound`.
2. The legacy default `n_subcyc` = 24 satisfies the bound too -- this is why
   the 100-yr mode-split runs never blew up:
   `test_legacy_default_is_inside_the_bound`.
3. The pre-fix dy-only formula VIOLATES the bound at the same geometry, so it
   cannot come back: `test_dy_only_sizing_violates_the_bound`.
4. The count scales with the drive (nu_h, dt), with the margin, and falls back
   to the meridional term when the zonal spacing is no longer the worst one:
   `test_count_scales_with_the_drive`.

Run:  python -m pytest tests/test_nu_nsub_cfl.py -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from config import R_EARTH
from jax_solver_global import nu_nsub_for_2d_cfl

# Production geometry, as make_global_grid builds it for the 1-deg runs:
# lat = linspace(-59.5, 59.5, 120), dlat = dlon = 1 deg, dy = R*radians(1).
NX, NY = 360, 120
DLAT = DLON = 1.0
NU_H = 5.0e6          # PhysicsConfig default
DT = 3600.0           # baroclinic step of the split runs
DT_BT = 150.0         # -> n_subcyc = 24
N_SUBCYC = int(round(DT / DT_BT))
MARGIN = 0.25         # nu_nsub_for_2d_cfl default
FTCS_BOUND = 0.5      # 5-point explicit Laplacian, per sub-step


def _metrics():
    """(dx_2d, dy) in metres for the production grid."""
    lat = np.linspace(-(NY / 2.0 - 0.5) * DLAT, (NY / 2.0 - 0.5) * DLAT, NY)
    dy = R_EARTH * np.radians(DLAT)
    dx_2d = np.broadcast_to(
        R_EARTH * np.radians(DLON) * np.cos(np.radians(lat)), (NX, NY)).copy()
    return dx_2d, dy


def _ftcs(nu_h, dt, dx_2d, dy, n):
    """nu_h * dt_sub * (1/dx^2 + 1/dy^2) at the smallest dx in the grid."""
    inv2 = float(np.max(np.asarray(dx_2d) ** -2)) + 1.0 / float(dy) ** 2
    return nu_h * (dt / (2.0 * n)) * inv2


def _dy_only_count(nu_h, dt, dy):
    """The pre-fix sizing: dy as a scalar, nothing else."""
    return max(1, int(np.ceil(nu_h * dt / (MARGIN * dy ** 2))))


def test_cfl_count_satisfies_the_2d_bound():
    dx_2d, dy = _metrics()
    n = nu_nsub_for_2d_cfl(NU_H, DT, dx_2d, dy)
    lhs = _ftcs(NU_H, DT, dx_2d, dy, n)
    print(f"[1] 1 deg grid: n={n} LHS={lhs:.4f} (margin {MARGIN})")
    assert n == 15, "the 1-deg production metric should size to 15 substeps"
    assert lhs <= MARGIN, (
        "the returned count does not hold the FTCS bound at its own margin")
    assert _ftcs(NU_H, DT, dx_2d, dy, n - 1) > MARGIN, (
        "one fewer substep still fits the bound: the count is not minimal")


def test_legacy_default_is_inside_the_bound():
    """n_subcyc = 24 is what every long run used; it must stay legal."""
    dx_2d, dy = _metrics()
    lhs = _ftcs(NU_H, DT, dx_2d, dy, N_SUBCYC)
    print(f"[2] legacy default n={N_SUBCYC}: LHS={lhs:.4f} (bound {FTCS_BOUND})")
    assert N_SUBCYC == 24
    assert lhs <= FTCS_BOUND, "the legacy default is over the FTCS bound"


def test_dy_only_sizing_violates_the_bound():
    """Pin the bug: sizing from dy alone lands ABOVE the bound at 1 deg."""
    dx_2d, dy = _metrics()
    n_old = _dy_only_count(NU_H, DT, dy)
    lhs_old = _ftcs(NU_H, DT, dx_2d, dy, n_old)
    print(f"[3] pre-fix dy-only: n={n_old} LHS={lhs_old:.4f} (bound {FTCS_BOUND})")
    assert n_old == 6
    assert lhs_old > FTCS_BOUND, (
        "the dy-only formula fits the bound after all: if that is true the "
        "2-D correction is unnecessary and this test should go")
    assert lhs_old / MARGIN > 2.0, (
        "the violation should be gross -- it is a missing metric term, not a "
        "marginal rounding")


def test_count_scales_with_the_drive():
    dx_2d, dy = _metrics()
    base = nu_nsub_for_2d_cfl(NU_H, DT, dx_2d, dy)
    assert nu_nsub_for_2d_cfl(2.0 * NU_H, DT, dx_2d, dy) > base
    assert nu_nsub_for_2d_cfl(NU_H, 0.5 * DT, dx_2d, dy) < base
    # Tighter margin -> more subcycles; looser -> fewer.
    assert nu_nsub_for_2d_cfl(NU_H, DT, dx_2d, dy, margin=0.125) > base
    assert nu_nsub_for_2d_cfl(NU_H, DT, dx_2d, dy, margin=0.5) < base
    # With the zonal spacing far coarser than dy, the meridional term is the
    # binding one: n = ceil(nu_h*dt/2/(dy^2*margin)) = 3.
    assert nu_nsub_for_2d_cfl(NU_H, DT, np.full((NX, NY), 10.0 * dy), dy) == 3
