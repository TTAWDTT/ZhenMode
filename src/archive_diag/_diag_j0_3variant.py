"""3-variant probe (god greenlight 2026-08-27T01-20): isolate what drives the
j=0 pole-wall explosion with the polar cap OFF.

god's framing: the cap-ON d1-NaN and cap-OFF d9-blowup are the SAME j=0
explosion; the cap just smears it faster (step 12 vs day 9). So the ROOT is
the j=0 explosion itself, and the cap is downstream. This probe isolates
which factor drives j=0 to blow up, with cap OFF, by tracking the EARLY
per-step growth of max|T| at j=0 (the explosion's nucleation) across:

  V0 baseline:             cap OFF, full metric, full dealias            (control)
  V1 zero metric@wall:     zero d2u_dx2 + corr at j=0,119 only          -> tests (i)
  V2 skip dealias@wall:    _dealias_h_fd leaves j=0,119 unfiltered      -> tests (ii)
  V3 both:                 V1 + V2                                       -> tests (iii)

Whichever variant kills the j=0 growth names the loop. ~200 steps is enough
to see the nucleation (the cap-ON version hit 1e+299 by step 12; cap-OFF is
slower but the early growth rate still distinguishes the driver).

Implementation: monkey-patches jax_solver_global module functions per variant
(so the JIT closure picks up the patched operator), rebuilds the solver each
time. No production code change.
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
from forcing import air_temp_profile, heat_flux_meridional
from wind_reanalysis import real_wind_forcing
from woa_data import get_initial_fields

# Build shared grid/forcing once (immutable inputs).
bathy = DEFAULT_CONFIG.bathymetry_file
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, bathy, smooth_passes=30, min_depth=100.0)
physics = replace(PhysicsConfig(), nu_h=5e6, nu_bi=0, kappa_bi=0)
Q_heat = heat_flux_meridional(grid, Q0=50.0)
T_init, S_init = get_initial_fields(grid)
T_init = np.array(T_init); S_init = np.array(S_init)
tau_x, tau_y = real_wind_forcing(month_idx=(2023 - 1948) * 12, grid=grid)
T_atm = air_temp_profile(grid, T_init[:, :, 0])
NY = grid.ny
WALL_ROWS = (0, NY - 1)


def run_variant(label, patch_fn):
    """Patch the module operators, rebuild solver, step 200x, track j=0 growth."""
    import jax_solver_global as g
    # Save originals so we can restore.
    orig_laplacian = g._laplacian_h
    orig_dealias = g._dealias_h_fd
    patch_fn(g, orig_laplacian, orig_dealias)
    try:
        # Re-import make_solver_global fresh (it closes over the module-level
        # operators via _compute_*; patched module funcs are picked up).
        step, init_state_fn, _ = g.make_solver_global(
            grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat),
            eos_type='linear', T_atm=T_atm, lambda_bulk=40.0,
            sponge_days=0.0, sponge_cells=0,
            T_init=T_init, S_init=S_init,
            polar_cap_rows=0, polar_cap_taper=0)
        state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
        print(f"\n=== {label} ===")
        print(f"{'stp':>4} {'max|T|j0':>12} {'max|T|int':>12} {'max|eta|j0':>12} {'Tj0_growth':>11}")
        prev_tj0 = 1e-30
        for k in range(200):
            state = step(state)
            T = np.array(state.T); eta = np.array(state.eta)
            if not np.isfinite(T).all():
                print(f"{k+1:>4}  NaN — j=0 exploded")
                return
            t_j0 = float(np.max(np.abs(T[:, 0, :])))
            t_int = float(np.max(np.abs(T[:, 10:110, :])))
            e_j0 = float(np.max(np.abs(eta[:, 0])))
            growth = t_j0 / prev_tj0 if prev_tj0 > 1e-30 else float('inf')
            if (k + 1) % 20 == 0 or k < 3:
                print(f"{k+1:>4} {t_j0:>12.4e} {t_int:>12.4e} {e_j0:>12.4e} {growth:>11.4f}")
            prev_tj0 = t_j0
            if t_j0 > 1e6:
                print(f"{k+1:>4} {t_j0:>12.4e}  — j=0 blowing up (growth {growth:.3f})")
                return
        print(f"  -> 200 steps, final max|T|j0 = {prev_tj0:.4e} (no blowup)")
    finally:
        # Restore originals.
        g._laplacian_h = orig_laplacian
        g._dealias_h_fd = orig_dealias


# --- Variant 0: baseline (no patch) ---
def patch_v0(g, orig_lap, orig_deal):
    g._laplacian_h = orig_lap
    g._dealias_h_fd = orig_deal


# --- Variant 1: zero the 1/cos^2 metric term (d2u_dx2 + corr) at wall rows ---
def patch_v1(g, orig_lap, orig_deal):
    def lap_zero_wall(u, p):
        lap = orig_lap(u, p)
        # Zero the zonal-2nd-derivative + metric-correction contribution at
        # the wall rows by zeroing the whole Laplacian there (the d2u_dy2 at
        # the wall is (u1-u0)/dy^2, a stable one-sided term we ALSO zero to
        # cleanly isolate the metric operator; if V1 kills growth, we refine).
        mask = jnp.ones((p.nx, p.ny, *([1] * (u.ndim - 2))))
        # build a (nx,ny,1...) mask with 0 at wall rows
        m = np.ones((p.nx, p.ny))
        m[:, list(WALL_ROWS)] = 0.0
        shp = [p.nx, p.ny] + [1] * (u.ndim - 2)
        mask = jnp.array(m.reshape(shp))
        return lap * mask
    g._laplacian_h = lap_zero_wall
    g._dealias_h_fd = orig_deal


# --- Variant 2: skip dealias at the wall rows ---
def patch_v2(g, orig_lap, orig_deal):
    def dealias_skip_wall(field, p):
        out = orig_deal(field, p)
        # Restore the un-dealiased field at wall rows (leave them as-is).
        m = np.ones((p.nx, p.ny))
        m[:, list(WALL_ROWS)] = 0.0
        shp = [p.nx, p.ny] + [1] * (field.ndim - 2)
        keep = jnp.array(m.reshape(shp))
        return keep * field + (1.0 - keep) * out
    g._laplacian_h = orig_lap
    g._dealias_h_fd = dealias_skip_wall


# --- Variant 3: both V1 + V2 ---
def patch_v3(g, orig_lap, orig_deal):
    patch_v1(g, orig_lap, orig_deal)
    dealias_skip_wall = g._dealias_h_fd  # currently orig (V1 didn't touch dealias)
    # re-apply V2 on top
    def dealias_skip_wall2(field, p):
        out = dealias_skip_wall(field, p)
        m = np.ones((p.nx, p.ny)); m[:, list(WALL_ROWS)] = 0.0
        shp = [p.nx, p.ny] + [1] * (field.ndim - 2)
        keep = jnp.array(m.reshape(shp))
        return keep * field + (1.0 - keep) * out
    g._dealias_h_fd = dealias_skip_wall2


if __name__ == '__main__':
    print(f"grid {grid.nx}x{grid.ny}, lat[{grid.lat[0]:.1f},{grid.lat[-1]:.1f}], "
          f"cap OFF, 200-step probe, tracking max|T| at j=0 vs interior")
    run_variant("V0 baseline (cap off, full metric, full dealias)", patch_v0)
    run_variant("V1 zero metric Laplacian @ wall rows (j=0,119)", patch_v1)
    run_variant("V2 skip dealias @ wall rows (j=0,119)", patch_v2)
    run_variant("V3 both V1+V2", patch_v3)

# --- Sanity check: cap-ON should explode at ~step 12 (confirms probe detects it) ---
def run_capon_sanity():
    import jax_solver_global as g
    g._laplacian_h = g._laplacian_h  # unpatched (module already restored)
    g._dealias_h_fd = g._dealias_h_fd
    step, init_state_fn, _ = g.make_solver_global(
        grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat),
        eos_type='linear', T_atm=T_atm, lambda_bulk=40.0,
        sponge_days=0.0, sponge_cells=0,
        T_init=T_init, S_init=S_init,
        polar_cap_rows=2, polar_cap_taper=3)  # cap ON
    state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
    print("\n=== SANITY: cap ON (polcap=2, taper=3) — should explode ~step 12 ===")
    for k in range(20):
        state = step(state)
        T = np.array(state.T)
        t_j0 = float(np.max(np.abs(T[:, 0, :]))) if np.isfinite(T).all() else float('nan')
        print(f"  step {k+1:>2}: max|T|j0 = {t_j0:.4e}")
        if not np.isfinite(T).all():
            print("  -> NaN (cap-ON explosion confirmed)"); break

if __name__ == '__main__' and 'SANITY' in os.environ:
    run_capon_sanity()
