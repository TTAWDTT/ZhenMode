"""DECISIVE VARIABLE-ISOLATION TEST: all-wet (water-flat) FD grid.

The spectral no-adv solver reaches bounded oscillatory equilibrium (max|u|
0.49 -> 4.13 -> 0.97, E bounded) while the FD no-adv solver grows unboundedly
(max|u| 2.81 -> 14.8, E 3.90e7 -> 1.08e8). Both carry the depth-varying
baroclinic PGF in the explicit residual; both start from rest; both have
advection off. Two differences confound the comparison:

  (1) linear-step exactness: spectral = matrix exponential; FD = FB Sielecki
  (2) land mask:             spectral = none (doubly-periodic, flat bottom);
                                FD = real wet_mask (coastlines)

The coastline-locality diagnostic (_diag_inject_locality, commit c82479d)
found 60.8% of the no-adv baroclinic KE in a 5-cell coastline buffer,
implicating the mask discontinuity (centered PGF stencil reaches into land
zeros at the coast). This test isolates variable (2): run the FD solver on
an ALL-WET grid (wet_mask = 1 everywhere, wet_mask_3d = 1 everywhere) with
the SAME real stratified T_init, SAME FB Sielecki linear step, advection OFF.

  If injection STOPS (bounded)  -> the land mask (coastline PGF discontinuity)
                                    is the sole source; the FB Sielecki linear
                                    step is fine. Fix = repair the coastal PGF.
  If injection CONTINUES (grows) -> the FB Sielecki linear step itself is the
                                    source (independent of mask). Fix = exact/
                                    implicit linear step (the confirmed direction).

depth is NOT read by make_fd_params (only wet_mask/wet_mask_3d), so overriding
the masks alone gives a true all-wet solver with the real dz layer structure.
"""
import os, sys
os.environ.setdefault('JAX_ENABLE_X64', '1')
sys.path.insert(0, 'src')
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import replace
from config import DEFAULT_CONFIG, PhysicsConfig, GlobalGridConfig
from grid import make_global_grid
import jax_solver_global as G
from jax_solver_global import (make_solver_global, JaxStateG, RHO_0, G_EARTH,
                               _step_impl)
from forcing import air_temp_profile, heat_flux_meridional

bathy = DEFAULT_CONFIG.bathymetry_file
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, bathy, smooth_passes=30, min_depth=100.0)

# ── Override: all-wet grid (no land mask), keep real dz stratification ──
nx, ny, nz = grid.nx, grid.ny, grid.nz
grid = replace(
    grid,
    wet_mask=np.ones((nx, ny), dtype=np.float64),
    ocean_mask=np.ones((nx, ny), dtype=bool),
    land_mask=np.zeros((nx, ny), dtype=bool),
    wet_mask_3d=np.ones((nx, ny, nz), dtype=np.float64),
)

Q_heat = heat_flux_meridional(grid, Q0=0.0)
d = np.load('src/_cache_init.npz')
T_init = d['T']; S_init = d['S']
# Land cells in T_init carry 0 (WOA fill); on the all-wet grid those become
# physical 0C water at the coast -> a huge T gradient. Fill land cells with
# the zonal mean of wet cells at each level so the all-wet field is smooth
# (removes the coastal T cliff that would otherwise inject via PGF).
wm_orig = np.array(make_global_grid(gcfg, bathy, smooth_passes=30,
                                    min_depth=100.0).wet_mask) > 0.5
T_smooth = np.array(T_init, dtype=np.float64)
S_smooth = np.array(S_init, dtype=np.float64)
for k in range(nz):
    for j in range(ny):
        wet_col = wm_orig[:, j]
        if wet_col.any():
            T_smooth[~wet_col, j, k] = T_init[wet_col, j, k].mean()
            S_smooth[~wet_col, j, k] = S_init[wet_col, j, k].mean()
        else:
            T_smooth[:, j, k] = T_init[:, :, k].mean()
            S_smooth[:, j, k] = S_init[:, :, k].mean()
T_init = T_smooth; S_init = S_smooth

tau_x = np.zeros((nx, ny)); tau_y = np.zeros((nx, ny))
T_atm = air_temp_profile(grid, T_init[:, :, 0])

physics = replace(PhysicsConfig(), nu_h=1e3, nu_bi=0, kappa_bi=0, r_bot=1e-3)
step, init_state_fn, _, params = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0,
    return_params=True)

# Zero advection (same as _diag_noadv_decisive)
def _zero_adv_u(u, v, w, p):
    z = jnp.zeros_like(v); return z, z
def _zero_adv_s(T, u, v, w, p):
    return jnp.zeros_like(T)
G._advection_flux_form = _zero_adv_u
G._advection_scalar = _zero_adv_s

step_no_adv = jax.jit(lambda s: _step_impl(s, params))

# All-wet -> wet_mask is all ones; energy over full domain.
dz = np.array(grid.dz); dz_norm = dz / dz.sum(); H = 4000.0
dz_norm_j = jnp.array(dz_norm)
@jax.jit
def energy_j(state):
    e = state.eta; u = state.u; v = state.v
    ua = 0.5 * (u[..., :-1] + u[..., 1:])
    va = 0.5 * (v[..., :-1] + v[..., 1:])
    ubt = jnp.sum(ua * dz_norm_j, -1)
    vbt = jnp.sum(va * dz_norm_j, -1)
    return (0.5 * H * jnp.sum(ubt**2 + vbt**2)
            + 0.5 * G_EARTH * H * jnp.sum(e**2))

# Baroclinic KE: total KE - barotropic KE
dz_norm_3d = jnp.array(dz_norm).reshape(1, 1, -1)
@jax.jit
def ke_bc_j(state):
    u = state.u; v = state.v
    ua = 0.5 * (u[..., :-1] + u[..., 1:])
    va = 0.5 * (v[..., :-1] + v[..., 1:])
    ke_full = jnp.sum(0.5 * H * (ua**2 + va**2) * dz_norm_3d)
    ubt = jnp.sum(ua * dz_norm_j, -1)
    vbt = jnp.sum(va * dz_norm_j, -1)
    ke_bt = 0.5 * H * jnp.sum(ubt**2 + vbt**2)
    return ke_full - ke_bt

@jax.jit
def diag_j(state):
    return (jnp.max(jnp.abs(state.eta)), jnp.max(jnp.abs(state.u)),
            jnp.isfinite(state.eta).all())

def run(label, step_fn, n=1400):
    state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
    print(f"\n=== {label} ===")
    print(f"{'stp':>5} {'E':>12} {'max|eta|':>10} {'max|u|':>10} {'KE_bc':>12}")
    for k in range(n):
        state = step_fn(state)
        if (k + 1) in (200, 400, 600, 800, 1000, 1200, 1400):
            E = float(energy_j(state)); me, mu, fin = diag_j(state)
            kebc = float(ke_bc_j(state).block_until_ready())
            print(f"{k+1:>5} {E:>12.4e} {float(me):>10.4e} {float(mu):>10.4e} "
                  f"{kebc:>12.4e}")
            if not bool(fin):
                print(f"  NaN step {k+1}"); break

run("ALL-WET FD, NO advection (real stratification, no land mask, FB Sielecki)",
    step_no_adv)
print("\nDONE.")
