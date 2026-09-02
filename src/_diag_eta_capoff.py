"""Compare eta growth: polar cap OFF vs ON. Is the cap injecting mass (eta)?"""
import os, sys
os.environ.setdefault('JAX_ENABLE_X64', '1')
sys.path.insert(0, 'src')
import jax, jax.numpy as jnp, numpy as np
from dataclasses import replace
from config import DEFAULT_CONFIG, PhysicsConfig, GlobalGridConfig
from grid import make_global_grid
from jax_solver_global import make_solver_global
from forcing import air_temp_profile, heat_flux_meridional
from wind_reanalysis import real_wind_forcing
from woa_data import get_initial_fields

def build(polcap):
    bathy=DEFAULT_CONFIG.bathymetry_file
    gcfg=replace(GlobalGridConfig(),lat_max=60.0,ny=120)
    grid=make_global_grid(gcfg,bathy,smooth_passes=30,min_depth=100.0)
    physics=replace(PhysicsConfig(),nu_h=5e6,nu_bi=0,kappa_bi=0)
    Q_heat=heat_flux_meridional(grid,Q0=50.0)
    T_init,S_init=get_initial_fields(grid); T_init=np.array(T_init); S_init=np.array(S_init)
    tau_x,tau_y=real_wind_forcing(month_idx=(2023-1948)*12,grid=grid)
    T_atm=air_temp_profile(grid,T_init[:,:,0])
    step,init_state_fn,_=make_solver_global(
        grid,physics,60.0,forcing=(tau_x,tau_y,Q_heat),eos_type='linear',
        T_atm=T_atm,lambda_bulk=40.0,sponge_days=0.0,sponge_cells=0,
        T_init=T_init,S_init=S_init,polar_cap_rows=polcap,polar_cap_taper=3)
    return step, init_state_fn, T_init, S_init, grid

print("=== polar cap OFF (polcap=0) ===")
step,init_fn,Ti,Si,grid=build(0)
state=init_fn(T_init=jnp.array(Ti),S_init=jnp.array(Si))
wm=np.array(grid.wet_mask)
for k in range(300):
    state=step(state)
    if (k+1)%50==0:
        eta=np.array(state.eta)
        mx=float(np.nanmax(np.abs(eta)))
        # mass: sum of eta over wet points (should be ~0 if mass conserved)
        mass=float(np.sum(eta[wm>0.5]))
        print(f"  step {k+1:>4}: max|eta|={mx:.3e}  sum_eta(wet)={mass:.3e}")
        if not np.isfinite(eta).all(): break

print("=== polar cap ON (polcap=2, taper 3) ===")
step,init_fn,Ti,Si,grid=build(2)
state=init_fn(T_init=jnp.array(Ti),S_init=jnp.array(Si))
wm=np.array(grid.wet_mask)
for k in range(300):
    state=step(state)
    if (k+1)%50==0:
        eta=np.array(state.eta)
        mx=float(np.nanmax(np.abs(eta)))
        mass=float(np.sum(eta[wm>0.5]))
        print(f"  step {k+1:>4}: max|eta|={mx:.3e}  sum_eta(wet)={mass:.3e}")
        if not np.isfinite(eta).all(): break
