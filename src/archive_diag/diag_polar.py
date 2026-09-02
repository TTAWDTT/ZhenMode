"""Probe: is the polar-cap edge the blow-up nucleation?

Finding from diag_global_stab.py: max|u| always at lat=82.5 (north polar row
-3, iz=13 bottom), grows 0->6.6 m/s in 4 steps, INDEPENDENT of nu_h (100/2000/
1e4 all identical). The polar-cap filter only zonally-averages rows :2 / -2:
of the BAROTROPIC (ubt/vbt/eta) fields. The 3D u,v at row -3 (and beyond) are
NOT filtered and NOT consistent with the averaged barotropic mode -> spurious
baroclinic shear at the cap edge -> explosive.

This probe tests two fixes by monkeypatching _free_surface_step_fd and adding
a 3D velocity polar filter after the step:
  (A) widen cap to 3 rows,
  (B) also zonally filter the 3D u,v at the cap rows (kill the inconsistency).

If (A)+(B) arrests the early growth, the root cause is confirmed and the fix
goes into jax_solver_global.py properly.
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from config import DEFAULT_CONFIG
from grid import make_global_grid, GlobalGridConfig
from woa_data import get_initial_fields
import jax_solver_global as G
from jax_solver_global import (
    JaxStateG, _barotropic_velocity, _compute_bt_rho_pgf, _d_dx, _d_dy,
    G_EARTH, RHO_0,
)

BATHY = r"C:\Users\zhen.luo\Desktop\ETOPO_2022_v1_r3600x1800_surface.nc"
NCAP = 3   # widen cap from 2 -> 3 rows


def _free_surface_step_fd_wide(eta, u, v, p, F_rho_x=None, F_rho_y=None,
                                dt_half=None):
    """Patched free-surface step: cap = NCAP rows, AND filter 3D u,v at cap."""
    if dt_half is None:
        dt_half = p.dt / 2.0
    ubt, vbt = _barotropic_velocity(u, v, p)
    F_x = jnp.zeros_like(ubt)
    F_y = jnp.zeros_like(vbt)
    if F_rho_x is not None:
        F_x = F_x + F_rho_x
        F_y = F_y + F_rho_y
    F_x = F_x + p.tau_x_2d / (RHO_0 * p.H_sw)
    F_y = F_y + p.tau_y_2d / (RHO_0 * p.H_sw)
    div_bt = _d_dx(ubt[:, :, None], p)[:, :, 0] + _d_dy(vbt[:, :, None], p)[:, :, 0]
    grad_eta_x = _d_dx(eta[:, :, None], p)[:, :, 0]
    grad_eta_y = _d_dy(eta[:, :, None], p)[:, :, 0]
    eta_new = eta - dt_half * p.H_sw * div_bt
    ubt_new = ubt + dt_half * (-G_EARTH * grad_eta_x + F_x)
    vbt_new = vbt + dt_half * (-G_EARTH * grad_eta_y + F_y)
    eta_new = eta_new * p.wet_mask
    ubt_new = ubt_new * p.wet_mask
    vbt_new = vbt_new * p.wet_mask

    # Wider + 3D-consistent polar cap: zonally average NCAP rows for eta,
    # ubt, vbt AND the 3D u,v (the baroclinic part too).
    def cap2d(field, n):
        c = jnp.mean(field[:, :n], axis=1, keepdims=True)
        field = field.at[:, :n].set(jnp.broadcast_to(c, (p.nx, n)))
        c = jnp.mean(field[:, -n:], axis=1, keepdims=True)
        field = field.at[:, -n:].set(jnp.broadcast_to(c, (p.nx, n)))
        return field
    def cap3d(field, n):
        c = jnp.mean(field[:, :n, :], axis=1, keepdims=True)
        field = field.at[:, :n, :].set(jnp.broadcast_to(c, (p.nx, n, p.nz)))
        c = jnp.mean(field[:, -n:, :], axis=1, keepdims=True)
        field = field.at[:, -n:, :].set(jnp.broadcast_to(c, (p.nx, n, p.nz)))
        return field

    eta_new = cap2d(eta_new, NCAP)
    ubt_new = cap2d(ubt_new, NCAP)
    vbt_new = cap2d(vbt_new, NCAP)
    # Filter the full 3D velocity at the cap rows so baroclinic + barotropic
    # are consistent (no spurious shear at the cap edge).
    u_cap = cap3d(u, NCAP)
    v_cap = cap3d(v, NCAP)

    delta_ubt = (ubt_new - ubt)[:, :, None]
    delta_vbt = (vbt_new - vbt)[:, :, None]
    u_new = u_cap + delta_ubt
    v_new = v_cap + delta_vbt
    return eta_new, u_new, v_new


def main():
    print("Building real global grid...")
    grid = make_global_grid(GlobalGridConfig(), BATHY)
    T_init, S_init = get_initial_fields(grid)
    print(f"  grid {grid.nx}x{grid.ny}x{grid.nz}, T [{T_init.min():.2f},{T_init.max():.2f}]")

    # Monkeypatch the wide-cap free surface into the module.
    G._free_surface_step_fd = _free_surface_step_fd_wide

    phys = DEFAULT_CONFIG.physics
    step, init_state, _ = G.make_solver_global(grid, phys, 60.0, forcing=None,
                                                eos_type='linear')
    state = init_state(T_init, S_init)
    print(f"\n=== wide cap (NCAP={NCAP}) + 3D velocity filter ===")
    print(f"  step  0: max_T={float(jnp.max(jnp.abs(state.T))):.3f}")
    for k in range(1, 21):
        state = step(state)
        max_T = float(jnp.max(jnp.abs(state.T)))
        max_u = float(jnp.max(jnp.abs(state.u)))
        nan = int(jnp.sum(jnp.isnan(state.T)))
        loc = ""
        if np.isfinite(max_u) and max_u > 0.5:
            au = np.abs(np.array(state.u))
            _, iy, iz = np.unravel_index(np.argmax(au), au.shape)
            loc = f" @lat={grid.lat[iy]:.1f} iz={iz}"
        print(f"  step {k:3d}: max|u|={max_u:.3e} max_T={max_T:.3f} nan={nan}{loc}")
        if not np.isfinite(max_T) or max_T > 1e4:
            print("  DIVERGING"); break


if __name__ == "__main__":
    main()
