"""Localize the free-surface mass leak (god 02-18): WHERE does volume appear/disappear?

Replicates _free_surface_step_fd internals manually so we can inspect the
PRE-MASK divergence and eta, and test god's mask-after-update hypothesis.

god's hypothesis: mask applied AFTER the eta update discards volume changes at
land/boundary cells => mass leak. We test:
  (1) WHERE is div_bt nonzero? coastline vs interior vs pole rows?
  (2) Does sum(-dt*H_sw*div_bt * wet) explain the observed sum(d_eta*wet)?
  (3) Is the leak the centered (roll) divergence NOT telescoping on the masked
      domain — i.e. would a conservative flux-form divergence (fluxes zeroed at
      wet/dry interfaces) conserve mass?

A centered roll/edge-pad divergence of a MASKED velocity field does NOT
telescope: at a coastline wet cell, the stencil reaches into the land neighbor
(value 0), so du/dx != 0 there => spurious volume source/sink exactly at coast.
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
from jax_solver_global import make_solver_global, JaxStateG, RHO_0, G_EARTH
from forcing import air_temp_profile, heat_flux_meridional
from woa_data import get_initial_fields

bathy = DEFAULT_CONFIG.bathymetry_file
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, bathy, smooth_passes=30, min_depth=100.0)
physics = replace(PhysicsConfig(), nu_h=1e3, nu_bi=0, kappa_bi=0, r_bot=1e-5)
Q_heat = heat_flux_meridional(grid, Q0=0.0)
T_init, S_init = get_initial_fields(grid)
T_init = np.array(T_init); S_init = np.array(S_init)
tau_x = np.zeros((grid.nx, grid.ny)); tau_y = np.zeros((grid.nx, grid.ny))
T_atm = air_temp_profile(grid, T_init[:, :, 0])
step, init_state_fn, _ = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0)

wm = np.array(grid.wet_mask) > 0.5     # (nx, ny) bool
nx, ny = grid.nx, grid.ny

# Replicate the FD stencils the solver uses (numpy).
inv_dx = np.array(grid.inv_dx) if hasattr(grid, 'inv_dx') else None
inv_dy = np.array(grid.inv_dy) if hasattr(grid, 'inv_dy') else None
# Fall back to constructing from grid spacing if inv_dx/inv_dy not present.
if inv_dx is None:
    lat = np.array(grid.lat)
    R = 6371.0e3
    dx = R * np.cos(np.deg2rad(lat)) * np.deg2rad(1.0)   # (ny,)
    inv_dx = np.broadcast_to(1.0/dx, (nx, ny)).copy()
    dy = R * np.deg2rad(1.0)
    inv_dy = np.full((nx, ny), 1.0/dy)
else:
    inv_dx = np.broadcast_to(inv_dx, (nx, ny)).copy() if inv_dx.ndim == 1 else inv_dx
    inv_dy = np.broadcast_to(inv_dy, (nx, ny)).copy() if inv_dy.ndim == 1 else inv_dy


def d_dx(u):
    return (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) * (0.5 * inv_dx)


def d_dy(u):
    up = np.pad(u, ((0, 0), (1, 1)), mode='edge')
    return (up[:, 2:] - up[:, :-2]) * (0.5 * inv_dy)


# Location classes on wet points.
dry = ~wm
nbr_dry = np.zeros((nx, ny), dtype=int)
nbr_dry += np.roll(dry, 1, axis=0).astype(int) + np.roll(dry, -1, axis=0).astype(int)
nbr_dry[:, 1:] += dry[:, :-1].astype(int); nbr_dry[:, :-1] += dry[:, 1:].astype(int)
is_coast = wm & (nbr_dry >= 1)
is_boundary = np.zeros((nx, ny), dtype=bool); is_boundary[:, 0] = True; is_boundary[:, -1] = True
is_interior = wm & ~is_coast & ~is_boundary
is_pole_near = np.zeros((nx, ny), dtype=bool); is_pole_near[:, :4] = True; is_pole_near[:, -4:] = True

print(f"grid {nx}x{ny}  wet={wm.sum()} coast={is_coast.sum()} interior={is_interior.sum()} "
      f"bndry_wet={(wm&is_boundary).sum()} pole_near_wet={(wm&is_pole_near).sum()}")

state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
e0 = np.array(state.eta)
yy = np.arange(ny); e0[:, yy] += 0.1 * np.sin(np.pi * yy / ny)   # 0.1m seiche bump
state = JaxStateG(state.u, state.v, state.T, state.S, jnp.array(e0))

H_sw = 4000.0
dt_half = 30.0

print("\n=== Divergence localization (after spinning up PGF velocity, pure free wave) ===")
# Step forward a few steps so the eta-bump pressure gradient has spun up velocity
# (step-1 div is zero because initial velocity is zero).
for _ in range(4):
    state = step(state)
print(f"  after 4 steps: max|ubt|={float(np.max(np.abs(np.array(state.u)))):.4e} "
      f"max|eta|={float(np.max(np.abs(np.array(state.eta)))):.4e}")
u = np.array(state.u); v = np.array(state.v)
# barotropic velocity (depth-avg). dz_norm sums to 1.
dz = np.array(grid.dz); dz_norm = dz / dz.sum()
u_avg = 0.5 * (u[..., :-1] + u[..., 1:])
v_avg = 0.5 * (v[..., :-1] + v[..., 1:])
ubt = np.sum(u_avg * dz_norm, axis=-1)
vbt = np.sum(v_avg * dz_norm, axis=-1)
# mask (solver masks ubt/vbt via wet_mask at the end of prev step; replicate)
ubt = ubt * wm; vbt = vbt * wm

div_bt = d_dx(ubt) + d_dy(vbt)     # RAW centered divergence (what solver uses)
eta = np.array(state.eta)
eta_new_raw = eta - dt_half * H_sw * div_bt
eta_new_masked = eta_new_raw * wm
d_eta = eta_new_masked - eta

def cls(mask, label, field):
    m = wm & mask
    s = float(np.sum(field[m]))
    mx = float(np.max(np.abs(field[m]))) if m.any() else 0.0
    return f"{label}: sum={s:+.4e} max|={mx:.4e}"

print("\n-- div_bt (raw centered, what the solver feeds to eta update) --")
print(f"  total wet:    sum={float(np.sum(div_bt[wm])):+.4e}")
print(f"  " + cls(is_interior, "interior ", div_bt))
print(f"  " + cls(is_coast,    "coastline", div_bt))
print(f"  " + cls(is_boundary, "boundary ", div_bt))
print(f"  " + cls(is_pole_near,"pole-near", div_bt))
print(f"  " + cls(~wm,        "land(dry)", div_bt))

print("\n-- d_eta after mask (eta_new*wm - eta) --")
print(f"  total wet:    sum={float(np.sum(d_eta[wm])):+.4e}")
print(f"  " + cls(is_interior, "interior ", d_eta))
print(f"  " + cls(is_coast,    "coastline", d_eta))
print(f"  " + cls(is_boundary, "boundary ", d_eta))
print(f"  " + cls(is_pole_near,"pole-near", d_eta))

# The discarded volume: what the mask threw away at land cells.
discarded = (eta_new_raw - eta_new_masked)
print(f"\n-- volume DISCARDED by *=wet_mask (eta_new_raw - eta_new_raw*wm) --")
print(f"  sum(land) = {float(np.sum(discarded[~wm])):+.4e}  (this is volume created/destroyed and dropped)")

# --- TEST: conservative flux-form divergence (fluxes zeroed at wet/dry iface) ---
# Face fluxes: U-face at (i+1/2, j) between col i and i+1; V-face at (i, j+1/2).
# A face is "open" only if BOTH neighbors are wet; else flux=0 (no flow into land).
ubt_w = ubt * wm; vbt_w = vbt * wm
# U-face flux = average of the two cell velocities, but zero if either side is land.
uface_open = wm & np.roll(wm, -1, axis=0)           # (i,j) wet AND (i+1,j) wet
uface = 0.5 * (ubt_w + np.roll(ubt_w, -1, axis=0)) * uface_open
vface_open = wm & np.roll(wm, -1, axis=1)
vface = 0.5 * (vbt_w + np.roll(vbt_w, -1, axis=1)) * vface_open
# Conservative divergence: flux out minus flux in.
div_cons = (uface - np.roll(uface, 1, axis=0)) + (vface - np.roll(vface, 1, axis=1))
# normalize by cell face area? The solver's div uses inv_dx/inv_dy 0.5*central.
# central div = (u_{i+1}-u_{i-1})/(2dx) = [0.5(u_i+u_{i+1}) - 0.5(u_{i-1}+u_i)]/dx
#            = (uface_i - uface_{i-1})/dx  when both faces open.
# So div_cons with /dx,/dy matches central EXCEPT at coastlines where faces close.
div_cons = div_cons * 1.0   # uface already carries the velocity; need /dx scaling:
# Actually central: (u_{i+1}-u_{i-1})/(2 dx). Flux form: (F_{i+1/2}-F_{i-1/2})/dx
# with F_{i+1/2}=0.5(u_i+u_{i+1}). These are EQUAL pointwise when both faces open.
# So scale div_cons by inv_dx (lon) and inv_dy (lat) appropriately:
div_cons_x = (uface - np.roll(uface, 1, axis=0)) * inv_dx
div_cons_y = (vface - np.roll(vface, 1, axis=1)) * inv_dy
div_cons = div_cons_x + div_cons_y

print("\n-- conservative flux-form divergence (faces closed at wet/dry boundary) --")
print(f"  total wet:    sum={float(np.sum(div_cons[wm])):+.4e}  (should be ~0 if mass-conserving)")
print(f"  " + cls(is_interior, "interior ", div_cons))
print(f"  " + cls(is_coast,    "coastline", div_cons))
eta_new_cons = (eta - dt_half * H_sw * div_cons) * wm
d_eta_cons = eta_new_cons - eta
print(f"\n-- d_eta with CONSERVATIVE divergence --")
print(f"  total wet:    sum={float(np.sum(d_eta_cons[wm])):+.4e}  (vs centered: {float(np.sum(d_eta[wm])):+.4e})")

# Multi-step: does the conservative divergence keep mass bounded?
print("\n=== Multi-step: centered (solver) vs conservative divergence ===")
print(f"{'stp':>4} {'sum_eta_ctr':>13} {'sum_eta_cons':>13} {'max_eta_ctr':>11} {'max_eta_cons':>11}")
state_c = JaxStateG(state.u, state.v, state.T, state.S, jnp.array(e0))
state_k = JaxStateG(state.u, state.v, state.T, state.S, jnp.array(e0))
for k in range(200):
    # centered (real solver)
    state_c = step(state_c)
    ec = np.array(state_c.eta)
    # conservative (manual replay using the SAME ubt/vbt the solver would form)
    # We can't easily replay the full 3D step manually, so just track the
    # divergence choice on the barotropic sub-step using the solver's own u,v.
    u3 = np.array(state_k.u); v3 = np.array(state_k.v)
    ua = 0.5*(u3[..., :-1]+u3[..., 1:]); va = 0.5*(v3[..., :-1]+v3[..., 1:])
    ub = np.sum(ua*dz_norm, axis=-1)*wm; vb = np.sum(va*dz_norm, axis=-1)*wm
    uf_open = wm & np.roll(wm, -1, axis=0); vf_open = wm & np.roll(wm, -1, axis=1)
    uf = 0.5*(ub+np.roll(ub,-1,axis=0))*uf_open
    vf = 0.5*(vb+np.roll(vb,-1,axis=1))*vf_open
    dc = (uf-np.roll(uf,1,axis=0))*inv_dx + (vf-np.roll(vf,1,axis=1))*inv_dy
    ek = np.array(state_k.eta)
    ek_new = (ek - dt_half*H_sw*dc) * wm
    # crude: keep u,v frozen (free-wave eta-only) to isolate mass conservation
    state_k = JaxStateG(state_k.u, state_k.v, state_k.T, state_k.S, jnp.array(ek_new))
    if (k+1) % 20 == 0 or k < 3:
        print(f"{k+1:>4} {float(np.sum(ec[wm])):>13.4e} {float(np.sum(ek_new[wm])):>13.4e} "
              f"{float(np.max(np.abs(ec))):>11.4e} {float(np.max(np.abs(ek_new))):>11.4e}")
    if not (np.isfinite(ec).all() and np.isfinite(ek_new).all()):
        print("  NaN"); break
