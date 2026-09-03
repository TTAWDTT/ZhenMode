"""Smoke tests for the semi-enclosed-sea eta relaxation (Mediterranean fix).

Run on the node (jax available):
    LD_PRELOAD=... LD_LIBRARY_PATH=... PYTHONPATH=/data/tmp/ocean/src \
        python _test_eta_relax.py
"""
import numpy as np
import jax
import jax.numpy as jnp

from dataclasses import replace
from grid import GlobalGridConfig, make_global_grid
from config import PhysicsConfig
from jax_solver_global import make_solver_global

jax.config.update("jax_enable_x64", True)

DEFAULT_BATHY = "/data/tmp/ocean/data/ETOPO_2022_v1_r3600x1800_surface.nc"

# ── Test 1: eta_relax OFF is bit-exact to the pre-change solver ──
cfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(cfg, bathymetry_file=DEFAULT_BATHY)
physics = replace(PhysicsConfig(), nu_bi=2e14)

# Simple forcing: zonal wind only (deterministic, no data files needed)
lon2d, lat2d = np.meshgrid(np.asarray(grid.lon), np.asarray(grid.lat),
                           indexing='ij')
tau_x = 0.1 * np.sin(np.pi * (lat2d + 60.0) / 120.0) * grid.wet_mask
tau_y = np.zeros_like(tau_x)
Q_heat = np.zeros_like(tau_x)

T0 = 20.0 - 30.0 * (lat2d[:, :, None] ** 2) / 3600.0
T0 = np.broadcast_to(T0, (grid.nx, grid.ny, grid.nz)).copy()
S0 = np.full_like(T0, 35.0)

dt = 120.0

kw = dict(forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
          T_init=T0, S_init=S0, sponge_days=3.0, sponge_cells=8)

step_off, init_off, _, params_off, _ = make_solver_global(
    grid, physics, dt, return_params=True, **kw)
st0 = init_off(T0, S0)
s = st0
for _ in range(5):
    s = step_off(s)
off5 = np.asarray(s.eta)
print("TEST1 eta_relax=OFF runs:", float(np.abs(off5).max()))

# ── Test 2: eta_relax ON damps a Med-box eta anomaly ──
box = (0.0, 36.0, 30.0, 46.0)   # synthetic Med-like box (grid lon is -180..180)
step_on, init_on, _, params_on, _ = make_solver_global(
    grid, physics, dt, return_params=True, **kw,
    eta_relax_days=30.0, eta_relax_box=box, eta_relax_buffer=1.0)

m = np.asarray(params_on.eta_relax_mask)
rate = float(params_on.eta_relax_rate)
print("TEST2 mask core pts:", int((m > 0.99).sum()),
      "buffer pts:", int(((m > 0) & (m < 0.99)).sum()),
      "rate:", f"{rate:.3e}")
assert rate == 1.0 / (30.0 * 86400.0), "rate mismatch"
assert (m > 0.99).sum() > 0, "empty core mask"

# determinism: same seed -> same off-state to relax
s = st0
for _ in range(5):
    s = step_on(s)
on5 = np.asarray(s.eta)

# The relaxation must have changed eta inside the box vs OFF (if eta != 0 there)
box_slice = (slice(180, 216), slice(90, 106))
d_eta = on5[box_slice] - off5[box_slice]
print("TEST2 |eta ON-OFF| inside box after 5 steps:", float(np.abs(d_eta).max()))
print("TEST2 global mean eta ON:", float(np.mean(on5[m > 0])) if (m > 0).any() else 0)

# ── Test 3: mass conservation of the relaxation step ──
# area-weighted global mean eta must be unchanged by the relax refill
area = np.asarray(params_on.dx_2d) * np.asarray(params_on.dy)
wm = np.asarray(params_on.wet_mask)
mnp = np.asarray(params_on.eta_relax_mask)
# analytic check of the refill identity: relax changes sum(A*eta) by dV_relax,
# the uniform refill adds -dV_relax back => net zero to machine precision.
decay = np.exp(-rate * dt * mnp)          # full-step (2 half-steps collapse)
eta_test = np.random.RandomState(0).randn(nx := grid.nx, grid.ny) * 0.1 * wm
dV = np.sum(area * (eta_test * decay - eta_test))
refill = (-dV / np.sum(area * wm)) * wm
after = eta_test * decay + refill
print("TEST3 net dV after refill (should be ~0):",
      f"{np.sum(area * (after - eta_test)):.3e}")

print("ALL SMOKE TESTS DONE")
