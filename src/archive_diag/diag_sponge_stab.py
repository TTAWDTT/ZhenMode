"""Wind-forced stability test for the global FD solver WITH polar-edge sponge.

G2 milestone gate: does the lateral sponge (Rayleigh damping at the polar
edge rows) arrest the wind-driven barotropic blow-up that pure Laplacian
dissipation can't (diverged at step 342 under nu_h=5e5)?

Setup (the config that held no-wind for 300 steps):
  lat_max=80 (ny=160), smooth_passes=20, min_depth=25,
  nu_h=5e5, nu_bi=0, dt=60s.

Wind: synthetic global zonal stress tau_x = tau0*sin(3*lat), tau0=0.1 Pa
(approximate observed mid-latitude westerly + tropical easterly structure).

Sponge: cosine-tapered over the poleward `sponge_cells` rows, relaxing
velocity/T/S/eta toward the WOA initial state (the "climatology"). This is
the global analogue of the regional solver's N/S-boundary sponge.

PASS gate: 1000+ steps, 0 NaN, max|T| bounded (< 40), max|u| bounded (< 2).

Usage:  cd src && export PYTHONPATH=/c/Users/zhen.luo/Python/Python314/site-packages \
        && /c/Python314/python.exe diag_sponge_stab.py
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import sys
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from dataclasses import replace

from config import DEFAULT_CONFIG, PhysicsConfig, GlobalGridConfig
from grid import make_global_grid
from woa_data import get_initial_fields
from jax_solver_global import make_solver_global

BATHY = r"C:\Users\zhen.luo\Desktop\ETOPO_2022_v1_r3600x1800_surface.nc"


def make_wind(grid, tau0=0.1):
    """Synthetic global zonal wind: tau_x = tau0*sin(3*lat), no meridional."""
    lat = np.array(grid.lat)            # (ny,)
    lat_rad = np.deg2rad(lat)
    tau_x = tau0 * np.sin(3.0 * lat_rad)   # (ny,)
    tau_x_2d = np.broadcast_to(tau_x[None, :], (grid.nx, grid.ny)).copy()
    tau_y_2d = np.zeros((grid.nx, grid.ny))
    Q_heat = np.zeros((grid.nx, grid.ny))
    return tau_x_2d, tau_y_2d, Q_heat


def run(grid, T_init, S_init, phys, dt, n_steps, tag,
        sponge_days=0.0, sponge_cells=0, wind=True, report_every=50, locate=False,
        polar_cap_rows=2):
    forcing = make_wind(grid) if wind else None
    step, init_state, _ = make_solver_global(
        grid, phys, dt, forcing=forcing, eos_type='linear',
        sponge_days=sponge_days, sponge_cells=sponge_cells,
        T_init=T_init, S_init=S_init, polar_cap_rows=polar_cap_rows)
    state = init_state(T_init, S_init)

    print(f"\n=== {tag} ===")
    print(f"  wind={wind}  sponge_days={sponge_days}  sponge_cells={sponge_cells}  "
          f"nu_h={phys.nu_h}  nu_bi={phys.nu_bi}  dt={dt}  cap_rows={polar_cap_rows}")
    print(f"  step   0: max|u|={float(jnp.max(jnp.abs(state.u))):.3e}  "
          f"max_T={float(jnp.max(jnp.abs(state.T))):.3f}  "
          f"nan={int(jnp.sum(jnp.isnan(state.T)))}")

    for k in range(1, n_steps + 1):
        state = step(state)
        max_T = float(jnp.max(jnp.abs(state.T)))
        max_u = float(jnp.max(jnp.abs(state.u)))
        nan = int(jnp.sum(jnp.isnan(state.T)))
        loc = ""
        if locate and np.isfinite(max_u) and max_u > 0.3:
            au = np.abs(np.array(state.u))
            ix, iy, iz = np.unravel_index(np.argmax(au), au.shape)
            lat = grid.lat[iy]; lon = grid.lon[ix]
            loc = f"  @({lat:+.1f},{lon:+.1f}) iz={iz}"
        if k % report_every == 0 or k <= 5 or not np.isfinite(max_T):
            print(f"  step {k:4d}: max|u|={max_u:.3e}  max_T={max_T:.3f}  nan={nan}{loc}")
        if not np.isfinite(max_T) or max_T > 1e4 or max_u > 1e3:
            print(f"  >>> DIVERGED at step {k}{loc}")
            return False, k
    ok = np.isfinite(max_T) and nan == 0 and max_T < 40.0 and max_u < 2.0
    print(f"  {'PASS' if ok else 'FAIL (held but out of bounds)'} at {n_steps} steps "
          f"(max|u|={max_u:.3e}, max_T={max_T:.3f}, nan={nan})")
    return ok, n_steps


def run_grid(lat_max, ny, smooth, min_depth, nu_h, tag, n_steps=1000,
             sponge_days=0.0, sponge_cells=0, wind=True, r_bot=None, cd=None,
             polar_cap_rows=2):
    print(f"\n{'='*60}\n{tag}: lat_max={lat_max} ny={ny} smooth={smooth} "
          f"min_depth={min_depth} nu_h={nu_h} r_bot={r_bot} cd={cd} cap={polar_cap_rows}\n{'='*60}")
    gcfg = replace(GlobalGridConfig(), lat_max=lat_max, ny=ny)
    grid = make_global_grid(gcfg, BATHY, smooth_passes=smooth, min_depth=min_depth)
    T_init, S_init = get_initial_fields(grid)
    T_init = np.array(T_init); S_init = np.array(S_init)
    print(f"  grid {grid.nx}x{grid.ny}x{grid.nz}, ocean {float(grid.wet_mask.mean()):.1%}")
    print(f"  T_init range [{T_init.min():.2f}, {T_init.max():.2f}]")
    kw = {}
    if r_bot is not None: kw['r_bot'] = r_bot
    if cd is not None: kw['cd'] = cd
    phys = replace(PhysicsConfig(), nu_h=nu_h, nu_bi=0.0, kappa_bi=0.0, **kw)
    run(grid, T_init, S_init, phys, 60.0, n_steps, tag,
        sponge_days=sponge_days, sponge_cells=sponge_cells, wind=wind,
        report_every=50, locate=True, polar_cap_rows=polar_cap_rows)


def main():
    # G2 Option-A gate: lat_max=60 (no metric singularity) + no-flux N/S wall
    # + nu_h=5e6 (spin-up stabilizer). god's gate = 1000 steps / 0 NaN /
    # max|T| bounded. Both no-wind AND wind-forced must hold (wind added per
    # the ordered attack: A stable no-wind -> THEN add wind).
    run_grid(60.0, 120, 30, 100.0, 5e6, "lat60 + wall, no wind (gate)",
             n_steps=1000, wind=False, polar_cap_rows=0)
    run_grid(60.0, 120, 30, 100.0, 5e6, "lat60 + wall, wind tau0=0.1 (gate)",
             n_steps=1000, wind=True, polar_cap_rows=0)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
