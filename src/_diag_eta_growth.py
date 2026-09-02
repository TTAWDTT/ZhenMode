"""Per-step eta growth at the north-wall hotspot, sponge=0, tapered cap on.
Is the eta growth exponential (instability) or linear (forced setup)?
Also check: is the polar cap creating an eta discontinuity at the cap edge?
"""
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
    T_init=T_init,S_init=S_init,polar_cap_rows=2,polar_cap_taper=3)
state=init_state_fn(T_init=jnp.array(T_init),S_init=jnp.array(S_init))
lat=grid.lat
print(f"{'stp':>4} {'max|eta|':>10} {'eta@Nwall':>10} {'eta@j114':>10} {'ratio':>8}")
prev=0.0
for k in range(60):
    state=step(state)
    eta=np.array(state.eta)
    mx=float(np.max(np.abs(eta)))
    # north wall region j=110..119
    nw=float(np.max(np.abs(eta[:,110:])))
    j114=float(np.max(np.abs(eta[:,114])))
    ratio = mx/prev if prev>1e-30 else float('inf')
    print(f"{k+1:>4} {mx:>10.4e} {nw:>10.4e} {j114:>10.4e} {ratio:>8.3f}")
    prev=mx
    if not np.isfinite(eta).all() or mx>1e6:
        print("  diverged"); break
