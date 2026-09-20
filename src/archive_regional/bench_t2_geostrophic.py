"""
Tier-2 test T2-2: Geostrophic adjustment (free-surface - Coriolis coupling).

Verifies that a localized free-surface SSH bump, released from rest on an
f-plane, radiates inertia-gravity waves and relaxes toward a steady state
in which the flow is in *geostrophic balance*:

    f0 * v = +g * d(eta)/dx      (meridional velocity balanced by zonal SSH slope)
    f0 * u = -g * d(eta)/dy      (zonal velocity balanced by meridional SSH slope)

This is the fundamental barotropic balance. It directly exercises the two
most important couplings in the solver that T2-1 could not test together:
the explicit barotropic pressure-gradient force (from the free surface eta)
and the exact Coriolis rotation. If either were mis-wired, the adjusted
flow would not satisfy geostrophy.

Setup (deliberately clean / analytic):
  - Place a Gaussian SSH bump eta0*exp(-(r/L)^2) in the center of a flat,
    uniform-density, CONSTANT-DEPTH (shallow-water) domain; u=v=0
    everywhere; no forcing; no bottom friction (isolate per T2-1).
  - Uniform T/S -> no baroclinic PGF; flat bottom -> no topographic effects.
  - The only physics active are free-surface PGF, Coriolis, and weak
    horizontal diffusion, so the adjustment is a pure barotropic geostrophic
    adjustment.
  - Propagate long enough (several wave traverse / inertial periods) for
    waves to radiate and the local eddy to settle.

Why the test uses a SHALLOW, uniform H_sw (not the production 4000 m):
  The production depth gives LR = sqrt(g*H_sw)/f0 ~ 2367 km, larger than the
  whole ~1400 km periodic domain. A sub-LR bump in a domain smaller than LR
  can never fully radiate its inertia-gravity waves (they re-enter through
  the periodic boundary), so the SSH never settles to a clean geostrophic
  state and the pointwise balance fails for regime reasons, not a solver
  bug. Shrinking H_sw (to ~100-200 m) brings LR (~370-530 km) inside the
  domain so a genuine adjustment can occur and the balance is verifiable.
  This is the same closure/depth isolation precedent as the inviscid T1d and
  zero-friction T2-1 checks: verify the operator coupling, not the turbulent
  ocean regime.

Verification:
  - After adjustment, derive the geostrophic velocity from the final eta via
    central differences in the interior and compare it to the actual final
    (u,v). An "adjusted" state is one where the RMS of the ageostrophic
    residual (u - u_geo, v - v_geo) is small relative to the velocity scale.
  - Metric: geostrophic correlation (mean of u- and v- correlations) and the
    ratio |residual|/|velocity| (RMS), evaluated on the interior away from
    the boundary/radiated wave zones.
  - Threshold: correlation > 0.95 and RMS residual ratio < 0.15.

Run (defaults tuned so LR fits the domain and the bump is resolvable):
  python src/bench_t2_geostrophic.py
  python src/bench_t2_geostrophic.py --days 4 --lbump 30   # quick diagnostic
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import replace

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from config import DEFAULT_CONFIG, G_EARTH
from grid import make_grid
from jax_solver import make_solver, JaxState

# ── Test parameters (defaults tuned: LR inside domain, resolvable bump) ─
ETA0 = 5.0          # m   peak SSH bump
H_SW = 200.0        # m   shallow-water depth (uniform) -> LR ~ 530 km
LBUMP = 35.0        #     bump e-folding radius in grid cells (~315 km)
DT = 50.0           # s   time step (gravity wave CFL-safe)
N_DAYS = 12.0       #     propagate ~12 days for waves to radiate
IOD = int(round(2.0 * np.pi / 7.2921e-5 / (24 * 3600)))  # ~24 h inertial period in days

CORR_MIN = 0.95      # min spatial correlation of (u,v) with geostrophic (u_geo,v_geo)
RATIO_MAX = 0.15     # max RMS ageostrophic residual / velocity scale
ACTIVE_FRAC = 0.10       # mask to points above 10% of the active-region max speed
SPONGE_TAU = 3600.0      # s  edge sponge damping timescale (0 = off); absorbs
                         #     wrap-around gravity waves on the periodic domain
SPONGE_FRAC = 0.15       #     outer fraction of each axis damped by the sponge


def _geo_velocity(eta, grid, f0):
    """Geostrophic velocity from SSH via central differences (interior only).

    Returns (u_geo, v_geo, mask) where mask is True on the interior strip
    (2..nx-3, 2..ny-3) used to avoid boundary/periodic artifacts.
    """
    nx, ny = eta.shape
    dx = float(grid.dx)
    dy = float(grid.dy)
    deta_dx = np.zeros_like(eta)
    deta_dy = np.zeros_like(eta)
    deta_dx[1:-1, 1:-1] = (eta[2:, 1:-1] - eta[:-2, 1:-1]) / (2.0 * dx)
    deta_dy[1:-1, 1:-1] = (eta[1:-1, 2:] - eta[1:-1, :-2]) / (2.0 * dy)
    v_geo = (G_EARTH / f0) * deta_dx
    u_geo = -(G_EARTH / f0) * deta_dy
    mask = np.zeros(eta.shape, dtype=bool)
    mask[2:-2, 2:-2] = True
    return u_geo, v_geo, mask


def _shallow_grid(h_sw, day_frac=0.25):
    """Grid copy with a uniform shallow depth so LR fits the periodic domain.

    Overrides the production (bathymetry-spanning) z/dz with a uniform
    profile of total depth `h_sw`, so the solver's H_sw = sum(dz) = h_sw and
    the barotropic deformation radius sqrt(g*H_sw)/f0 sits inside the domain.
    """
    g = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    nz = g.nz
    dz = np.full(nz - 1, h_sw / (nz - 1), dtype=np.float64)
    z = np.zeros(nz, dtype=np.float64)
    z[1:] = np.cumsum(dz)             # negative downward: z = [-0, -d1, ...]
    z = -z
    g.z = z
    g.dz = dz
    # Keep the bathymetry used for the mask; make H_mean derive from the
    # shallow depth for a self-consistent test (unused by the balance check).
    return g


def _sponge_step(step, dt, nx, ny, tau=SPONGE_TAU, frac=SPONGE_FRAC):
    """Wrap the solver step with a Newtonian (Rayleigh) sponge near the
    periodic-boundary edges.

    A periodic domain with no sponge cannot settle to geostrophy: gravity
    waves emitted by the bump wrap around and keep re-entering the interior
    forever, so the flow stays in a standing/sloshing regime (the diagnostic
    showed eta_peak oscillating non-monotonically and residual ratio ~2.8
    even after many inertial periods). Damping u, v and eta in a thin band
    near each edge absorbs outgoing waves before they wrap, letting the
    interior genuinely radiate its imbalance and settle to the geostrophic
    residual. The interior is left untouched (sponge == 0 there), so the
    measured balance is unaffected. `tau <= 0` disables the sponge.
    """
    y, x = np.mgrid[0:ny, 0:nx]
    dd = np.maximum(
        np.abs(x - (nx - 1) / 2.0),
        np.abs(y - (ny - 1) / 2.0),
    ) / np.maximum(nx, ny)                      # 0 at center, ~0.5 at edges
    ramp = np.clip((dd * 2.0 - (1.0 - frac)) / frac, 0.0, 1.0)
    sp2d = ramp * ramp                          # (ny,nx) taper, 1 at corners
    sp3d = sp2d[..., None]                      # broadcast over depth
    f = dt / tau if tau > 0 else 0.0

    def wrapped(state):
        state = step(state)
        if f > 0:
            state = state._replace(
                u=state.u * (1.0 - f * sp3d),
                v=state.v * (1.0 - f * sp3d),
                eta=state.eta * (1.0 - f * sp2d),
            )
        return state
    return wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=N_DAYS)
    ap.add_argument("--lbump", type=float, default=LBUMP)
    ap.add_argument("--hsw", type=float, default=H_SW)
    ap.add_argument("--sponge-tau", type=float, default=SPONGE_TAU,
                    help="edge sponge damping timescale in s (0 = off; internal diag uses small tau)")
    args = ap.parse_args()

    grid = _shallow_grid(args.hsw)
    # Isolate viscous closures (T1d precedent): the analytic geostrophic-
    # adjustment result is a clean free-surface + Coriolis balance. Use weak
    # Laplacian diffusion only (no biharmonic stabilizer) so the adjusted
    # velocity is not spurious-damped while the SSH slope stays sharp.
    physics = replace(
        DEFAULT_CONFIG.physics,
        bottom_friction='none', r_bot=0.0,
        nu_h=200.0, kappa_h=100.0, nu_bi=0.0, kappa_bi=0.0,
    )

    f0 = float(grid.f0)
    nx, ny = grid.nx, grid.ny
    lr = float(np.sqrt(G_EARTH * args.hsw) / f0)
    print("=" * 60)
    print("T2-2 GEOSTROPHIC ADJUSTMENT (free-surface + Coriolis)")
    print("=" * 60)
    print(f"grid {nx}x{ny}x{grid.nz}  f0={f0:.6e} s^-1  "
          f"f0^-1={1/f0/3600:.2f} h")
    print(f"H_sw={args.hsw:g} m -> LR=sqrt(gH)/f0 ~ {lr/1e3:.0f} km "
          f"(domain ~ {nx*float(grid.dx)/1e3:.0f} x {ny*float(grid.dy)/1e3:.0f} km)")

    step, init_state, _ = make_solver(grid, physics, DT)
    if args.sponge_tau > 0:
        step = _sponge_step(step, DT, nx, ny, tau=args.sponge_tau)
    base = init_state()
    print(f"edge sponge: {'ON tau=%.0f s' % args.sponge_tau if args.sponge_tau > 0 else 'OFF'} "
          f"(absorbs wrap-around gravity waves on the periodic domain)")

    # Gaussian SSH bump (scale by grid cell size in metres).
    dx = float(grid.dx)
    i0, j0 = nx // 2, ny // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    xs = (xx - i0) * dx
    ys = (yy - j0) * dx  # ~isotropic cells; use dx for both
    r2 = (xs ** 2 + ys ** 2) / (args.lbump * dx) ** 2
    eta0 = ETA0 * np.exp(-r2).astype(np.float64)

    state = JaxState(
        u=jnp.zeros_like(base.u),
        v=jnp.zeros_like(base.v),
        T=base.T,
        S=base.S,
        eta=jnp.asarray(eta0),
    )

    n_steps = int(round(args.days * 24 * 3600 / DT))
    print(f"\nseed: Gaussian SSH bump eta0={ETA0} m, L={args.lbump} cells "
          f"({args.lbump*dx/1e3:.0f} km), u=v=0, no forcing, no bottom friction")
    print(f"propagating {args.days:.0f} days ({n_steps} steps, dt={DT:.0f} s) "
          f"~ {args.days/IOD:.1f} inertial periods")

    t = 0.0
    anim = 0.0
    for n in range(n_steps):
        state = step(state)
        t += DT
        anim += DT
        # Light progress every ~24h of model time.
        if anim >= 24 * 3600:
            print(f"  t = {t/86400:6.1f} d   "
                  f"eta_peak = {float(jnp.max(jnp.abs(state.eta))):8.4f} m   "
                  f"vel_max = {float(jnp.max(jnp.hypot(state.u, state.v))):8.5f} m/s")
            anim = 0.0

    u = np.asarray(state.u)[:, :, 0]   # top level (barotropic limit of the setup)
    v = np.asarray(state.v)[:, :, 0]
    eta = np.asarray(state.eta)

    u_geo, v_geo, strip = _geo_velocity(eta, grid, f0)

    # The far field of a periodic, still-adjusting run holds weak, unbalanced
    # wave noise that drowns a correlation computed over the whole interior.
    # Geostrophy is physically meaningful where the flow is active, so we
    # restrict the metric to the dynamically active region: points whose local
    # velocity amplitude is above a small fraction of its interior max.
    speed = np.hypot(u, v)
    active = strip & (speed > ACTIVE_FRAC * speed[strip].max())

    def metrics(mask):
        u_i, v_i = u[mask], v[mask]
        ug_i, vg_i = u_geo[mask], v_geo[mask]
        vel_rms = float(np.sqrt(np.mean(u_i ** 2 + v_i ** 2)))
        res_rms = float(np.sqrt(np.mean((u_i - ug_i) ** 2 + (v_i - vg_i) ** 2)))
        ratio = res_rms / vel_rms if vel_rms > 0 else float('nan')

        def corr(a, b):
            a = a - a.mean()
            b = b - b.mean()
            den = np.sqrt((a * a).mean() * (b * b).mean())
            return float((a * b).mean() / den) if den > 0 else 0.0

        corr_all = 0.5 * (corr(u_i, ug_i) + corr(v_i, vg_i))
        return vel_rms, res_rms, ratio, corr_all

    vel_rms, res_rms, ratio, corr_all = metrics(active)
    _f_vel, _f_res, _f_ratio, _f_corr = metrics(strip)

    print("\nfinal state (top level, after adjustment):")
    print(f"  ACTIVE region ({active.sum()} pts):")
    print(f"    velocity RMS          = {vel_rms:.4f} m/s")
    print(f"    ageostrophic resid RMS= {res_rms:.4f} m/s")
    print(f"    residual ratio        = {ratio:.4f}")
    print(f"    geostrophic corr      = {corr_all:.4f}")
    print(f"  [full interior strip: vel_rms={_f_vel:.4f} res_rms={_f_res:.4f} "
          f"ratio={_f_ratio:.3f} corr={_f_corr:+.3f}]")

    ok = (corr_all >= CORR_MIN) and (ratio <= RATIO_MAX)
    print(f"\n{'PASS' if ok else 'FAIL'}: active-region corr {corr_all:.3f} "
          f">= {CORR_MIN:.2f} and ratio {ratio:.3f} <= {RATIO_MAX:.2f}")
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
