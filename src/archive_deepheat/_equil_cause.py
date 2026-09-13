"""What sets the model's (too warm) deep equilibrium?

The model's abyss equilibrates near 5-6 C while the real ocean is 1-2 C.
Hypothesis: lat_max=60 means the COLDEST water that can sink is the SST at
60N/60S, which the zonal-WOA restoring profile puts at ~5-8 C.  There is no
sea-ice / polar deep-water formation region, so the abyss can never be
colder than that.

Measure:
  A. T_atm(lat) profile and the area-weighted ocean mean (the SST the bulk
     flux drives the surface toward).
  B. heat_T (penetrating SW) - its column integral and spatial pattern.
  C. bulk_T at the ckpt - mean flux, and the implied surface heat input.
  D. where convection actually occurs (surface density maxima) at the ckpt.
"""
import sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
sys.path.insert(0, "src")
from dataclasses import replace
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
from forcing import air_temp_profile, BULK_LAMBDA_DEFAULT
import jax_solver_global as JS

g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
lat = np.asarray(g.lat, float) if hasattr(g, "lat") else np.linspace(-59.5, 59.5, 120)
init = np.load("init_fields_g360x120.npz")
T_atm_np = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])

print("=" * 96)
print("A. T_atm(lat)  -- the temperature the Haney bulk flux restores SST toward")
print("=" * 96)
Ta2d = np.asarray(T_atm_np, float)
if Ta2d.ndim == 3:
    Ta2d = Ta2d[:, :, 0]
wet2 = np.asarray(g.wet_mask, float) > 0.5
print("T_atm raw min %.2f max %.2f" % (Ta2d.min(), Ta2d.max()))
print("\n  lat      T_atm(zonal)   n_wet   ocean area frac")
for j in range(0, 120, 6):
    a = (AREA[:, j] * wet2[:, j]).sum()
    print("  %+6.1f      %6.2f        %4d      %.4f"
          % (lat[j], float(Ta2d[:, j].mean()), int(wet2[:, j].sum()),
             a / (AREA * wet2).sum()))
ocean_area = (AREA * wet2).sum()
Tatm_mean = float((Ta2d * AREA * wet2).sum() / ocean_area)
print("\n  AREA+OCEAN-WEIGHTED MEAN T_atm = %.3f C   <-- the SST the model targets" % Tatm_mean)
print("  real global-mean SST  = 17.0-18.0 C")
print("  real deep-water formation T = -1 to 2 C (at 70-80 deg lat, under ice)")
print("  T_atm at the poleward edge (lat=+/-59.5): %.2f C" % float(Ta2d[:, 0].mean()))

print("\n" + "=" * 96)
print("B/C. surface heat budget at the ten-year checkpoint")
print("=" * 96)
BASE = dict(nu_h=5e6, nu_bi=2e14, kappa_bi=2e14, kappa_gm=1000.0,
            kappa_redi=1000.0, kappa_v=1e-5, kappa_conv=0.05, gm_slope_max=0.005)
_, _, _, p, terms = JS.make_solver_global(
    g, replace(PhysicsConfig(), **BASE), 3600.0, T_atm=jnp.asarray(T_atm_np),
    lambda_bulk=BULK_LAMBDA_DEFAULT, mode_split=True, dt_bt=300.0, return_params=True)
wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
surf = np.asarray(p.surface_mask, float).ravel()
print("surface_mask shape", np.asarray(p.surface_mask).shape, "sum", surf.sum())
CP = 1025.0 * 3992.0

d = np.load("results/ckpt_tenyr_ms_gm.npz")
st = JS.JaxStateG(u=jnp.asarray(d["u"]), v=jnp.asarray(d["v"]),
                  T=jnp.asarray(d["T"]), S=jnp.asarray(d["S"]),
                  eta=jnp.asarray(d["eta"]))

# recompute the two surface-flux terms exactly as _compute_tracer_tendency does
import inspect
src = inspect.getsource(JS._compute_tracer_tendency)
for ln in src.splitlines():
    if "heat_T" in ln or "bulk" in ln.lower():
        print("   [src] " + ln.strip())
