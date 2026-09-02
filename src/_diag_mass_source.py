"""Isolate the mass leak: test 3 configs over 300 steps, track sum_eta(wet).
  A: cap OFF, no wind (tau=0) — pure free-surface step, should conserve mass exactly.
  B: cap OFF, with wind — does wind cause the leak?
  C: cap ON, no wind — does the cap leak?
This triangulates the mass source.
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
from woa_data import get_initial_fields

def build(polcap, wind_on):
    bathy=DEFAULT_CONFIG.bathymetry_file
    gcfg=replace(GlobalGridConfig(),lat_max=60.0,ny=120)
    grid=make_global_grid(gcfg,bathy,smooth_passes=30,min_depth=100.0)
    physics=replace(PhysicsConfig(),nu_h=5e6,nu_bi=0,kappa_bi=0)
    Q_heat=heat_flux_meridional(grid,Q0=50.0)
    T_init,S_init=get_initial_fields(grid); T_init=np.array(T_init); S_init=np.array(S_init)
    if wind_on:
        from wind_reanalysis import real_wind_forcing
        tau_x,tau_y=real_wind_forcing(month_idx=(2023-1948)*12,grid=grid)
    else:
        tau_x=np.zeros((grid.nx,grid.ny)); tau_y=np.zeros((grid.nx,grid.ny))
    T_atm=air_temp_profile(grid,T_init[:,:,0])
    step,init_state_fn,_=make_solver_global(
        grid,physics,60.0,forcing=(tau_x,tau_y,Q_heat),eos_type='linear',
        T_atm=T_atm,lambda_bulk=40.0,sponge_days=0.0,sponge_cells=0,
        T_init=T_init,S_init=S_init,polar_cap_rows=polcap,polar_cap_taper=3)
    return step, init_state_fn, T_init, S_init, grid

wm=None
for label,pc,wind in [("A: cap OFF, no wind",0,False),("B: cap OFF, wind",0,True),("C: cap ON, no wind",2,False)]:
    step,init_fn,Ti,Si,grid=build(pc,wind)
    if wm is None: wm=np.array(grid.wet_mask)
    state=init_fn(T_init=jnp.array(Ti),S_init=jnp.array(Si))
    # seed a 1m eta bump so we can see if it's conserved/damped/leaked
    state=jax.tree_util.tree_map(lambda x:x, state)
    e0=np.array(state.eta); e0[wm>0.5]+=1.0
    from jax_solver_global import JaxStateG
    state=JaxStateG(state.u,state.v,state.T,state.S,jnp.array(e0))
    print(f"=== {label} ===  (seeded +1m eta bump, sum_init={float(np.sum(e0[wm>0.5])):.1f})")
    for k in range(300):
        state=step(state)
        if (k+1)%100==0:
            eta=np.array(state.eta)
            mx=float(np.nanmax(np.abs(eta)))
            mass=float(np.sum(eta[wm>0.5]))
            print(f"  step {k+1:>4}: max|eta|={mx:.3e}  sum_eta(wet)={mass:.3e}")
            if not np.isfinite(eta).all(): break
