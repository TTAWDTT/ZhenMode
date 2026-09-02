"""Check geostrophic balance of the eta buildup at step 100 (config D, no forcing).

If eta is in geostrophic balance: f*ubt = -g*dy(eta), f*vbt = g*dx(eta).
Then eta reflects a REAL (if amplified) barotropic geostrophic circulation
driven by the baroclinic PGF — a physical adjustment, not a numerical blowup.
The fix would target the AMPLITUDE of the baroclinic-driven ubt.

If NOT geostrophic (eta grows while ubt doesn't balance it), it's a free-surface
instability — the FB coupling or cap injects eta independently.
"""
import os, sys
os.environ.setdefault('JAX_ENABLE_X64', '1')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.92')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
sys.path.insert(0, 'src')
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import replace

from config import DEFAULT_CONFIG, PhysicsConfig, GlobalGridConfig, G_EARTH, RHO_0
from grid import make_global_grid
import jax_solver_global as G
from jax_solver_global import make_solver_global, _barotropic_velocity, _gradient_conservative, _d_dx, _d_dy, JaxStateG
from woa_data import get_initial_fields
from run_long_integration_global import heat_flux_meridional, air_temp_profile

gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
ocean = np.asarray(grid.ocean_mask, dtype=bool)
H_sw = float(np.sum(grid.dz))

physics = replace(PhysicsConfig(), nu_h=5e6, nu_bi=0.0, kappa_bi=0.0)
Q_heat = heat_flux_meridional(grid, Q0=50.0)
T_init, S_init = get_initial_fields(grid)
T_init = np.array(T_init); S_init = np.array(S_init)
T_atm = air_temp_profile(grid, T_init[:, :, 0])

# config D: no wind, no F_rho in free surface
G._compute_bt_rho_pgf = lambda state, p: (jnp.zeros((p.nx, p.ny)), jnp.zeros((p.nx, p.ny)))
step, init_state, _, p = make_solver_global(
    grid, physics, 60.0, forcing=(np.zeros((grid.nx,grid.ny)), np.zeros((grid.nx,grid.ny)), Q_heat),
    eos_type='linear', T_atm=T_atm, lambda_bulk=40.0,
    sponge_days=3.0, sponge_cells=16, T_init=T_init, S_init=S_init,
    polar_cap_rows=2, polar_cap_taper=3, return_params=True)
step = jax.jit(step)
state = init_state(T_init, S_init)
for k in range(100):
    state = step(state)

eta = np.asarray(state.eta)
ubt_j, vbt_j = _barotropic_velocity(state.u, state.v, p)
ubt = np.asarray(ubt_j); vbt = np.asarray(vbt_j)
f2d = np.asarray(p.f)

# geostrophic: u_geo = -(g/f)*deta/dy, v_geo = (g/f)*deta/dx
grad_eta_x_j, grad_eta_y_j = _gradient_conservative(state.eta, p)
detax = np.asarray(grad_eta_x_j); detay = np.asarray(grad_eta_y_j)
u_geo = -G_EARTH * detay / f2d
v_geo =  G_EARTH * detax / f2d

print(f"step 100, config D (no wind, no F_rho): max|eta|={np.abs(eta).max():.2f}m max|ubt|={np.abs(ubt).max():.4f}")
print(f"max|u_geo|={np.abs(u_geo*ocean).max():.4f}  max|v_geo|={np.abs(v_geo*ocean).max():.4f}")
# correlation between ubt and u_geo over ocean
m = ocean.flatten()
ub = ubt.flatten()[m!=0]; ug = u_geo.flatten()[m!=0]
vb = vbt.flatten()[m!=0]; vg = v_geo.flatten()[m!=0]
def corr(a,b):
    a=a-a.mean(); b=b-b.mean()
    return (a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum())
print(f"corr(ubt, u_geo) = {corr(ub,ug):.4f}")
print(f"corr(vbt, v_geo) = {corr(vb,vg):.4f}")
print(f"ratio std(ubt)/std(u_geo) = {ub.std()/ug.std():.3f}")
print(f"ratio std(vbt)/std(v_geo) = {vb.std()/vg.std():.3f}")
# residual: f*ubt + g*detay (should be ~0 if geo)
res_u = f2d*ubt + G_EARTH*detay
res_v = f2d*vbt - G_EARTH*detax
print(f"ageostrophic residual max|f*u+g*detay|={np.abs(res_u*ocean).max():.2e}  max|f*v-g*detax|={np.abs(res_v*ocean).max():.2e}")
print(f"  compare |f*u| max={np.abs(f2d*ubt*ocean).max():.2e}  |g*detay| max={np.abs(G_EARTH*detay*ocean).max():.2e}")

# where is the eta max? coast / shelf / interior?
lat = np.asarray(grid.lat); lon = np.asarray(grid.lon)
amax = np.abs(eta*ocean)
ij = np.unravel_index(np.argmax(amax), eta.shape)
print(f"\neta max location: lon={lon[ij[0]]:.1f}E lat={lat[ij[1]]:.1f}  eta={eta[ij]:.2f}m")
# is it a coastline point? check neighbors in ocean_mask
om = ocean
nx, ny = eta.shape
i,j = ij
nbrs = [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]
land_nbr = sum(1 for (a,b) in nbrs if 0<=a<nx and 0<=b<ny and not om[a,b])
print(f"  land neighbors: {land_nbr}/4  (coastline if >0)")
# mass conservation: mean eta over ocean
mean_eta = (eta*ocean).sum()/ocean.sum()
print(f"mean eta (mass) = {mean_eta:.4f}m  (should be ~0 if mass-conserved)")
# distribution: how many points > 5m?
print(f"ocean pts |eta|>1m: {(np.abs(eta*ocean)>1).sum()}  >5m: {(np.abs(eta*ocean)>5).sum()}  >10m: {(np.abs(eta*ocean)>10).sum()}")
# div(ubt) at the hotspot
divbt_j = G._divergence_conservative(ubt_j*p.wet_mask, vbt_j*p.wet_mask, p)
divbt = np.asarray(divbt_j)
print(f"div(ubt) at hotspot: {divbt[ij]:.2e}/s   eta_incr/step = -dt*H_sw*div = {-60*H_sw*divbt[ij]:.4f}m")
# global: what is the typical eta growth rate spatially?
eta_incr_field = -60*H_sw*divbt
print(f"max |eta_incr/step| field = {np.abs(eta_incr_field*ocean).max():.4f}m  rms = {np.sqrt(np.mean((eta_incr_field*ocean)**2)):.4f}m")
