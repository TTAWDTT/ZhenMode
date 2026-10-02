"""Mechanically preserved FD stability implementation."""
from .backend import np


def nu_nsub_for_2d_cfl(nu_h, dt, dx_2d, dy, margin=0.25):
    """Subcycle count that keeps the explicit nu_h Laplacian inside its FTCS bound.

    The 5-point Laplacian's most negative eigenvalue sums BOTH metric terms, so
    the split L half-step is stable iff
    ``nu_h * dt_sub * (1/dx^2 + 1/dy^2) <= margin`` with ``dt_sub = dt/(2*n)``.
    The worst point is the ZONAL spacing at the highest latitude -- ``dx =
    dy*cos(lat)``, so at lat_max=60 the zonal spacing is HALF the meridional one
    and its term is 3.9x the meridional one in the sum. Sizing from ``dy`` alone
    (a scalar) misses that: at nu_h=5e6, dt=3600, lat_max=60 it returned 6. The
    bound above is deliberately conservative -- the REALIZED most negative
    eigenvalue of this stencil on the ETOPO metric is 1.4515e-9, 91.9% of the
    sum-of-worst-cases 1.5794e-9 -- so the growth threshold is nu_h*dts*|lam|
    > 2, i.e. LHS > 0.544, i.e. n >= 6.53 (7). The old formula's 6 sits 9% over
    that line: |1 + nu_h*dts*lam| = 1.177 per substep, 2.66x per half-step.
    This returns 15 (factor 0.13, i.e. strongly damped).

    The polar cap hides the symptom by zonally averaging exactly those rows, so
    the undersizing stays latent while the cap does its job: the cap takes the
    n=6 spectral radius from 2.66 to 1.00 (the strictly neutral modes), and
    mis-anchoring its north band leaves 1.53 -- the growth that was measured as
    FAIL_BLOWUP at step 36 with the peak pinned to j=ny-1. At the production
    count (n=24) the same radius is 1.00 with NO cap at all, which is why the
    100-yr runs were stable for their own reason and not by accident. (D21)

    Returns the count for the split path; the monolithic path applies nu_h once
    per half-step and is sized by dt=60-300 instead. (D12)
    """
    inv_dx2_max = float(np.max(np.asarray(dx_2d) ** -2))   # smallest dx wins
    inv_dy2 = 1.0 / float(dy) ** 2
    return max(1, int(np.ceil(nu_h * (dt / 2.0)
                               * (inv_dx2_max + inv_dy2) / margin)))
