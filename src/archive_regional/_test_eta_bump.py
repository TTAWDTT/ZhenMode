"""Direct validation: eta bump inside the relax box must decay at exp(-rate*t)."""
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from dataclasses import replace
from grid import GlobalGridConfig, make_global_grid
from config import PhysicsConfig
from jax_solver_global import make_solver_global, JaxStateG
import jax.numpy as jnp

BATHY = "/data/tmp/ocean/data/ETOPO_2022_v1_r3600x1800_surface.nc"
cfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(cfg, bathymetry_file=BATHY)
physics = replace(PhysicsConfig(), nu_bi=2e14)
nx, ny, nz = grid.nx, grid.ny, grid.nz
tau_x = np.zeros((nx, ny)); tau_y = np.zeros((nx, ny)); Q = np.zeros((nx, ny))
lat2 = np.asarray(grid.lat)
T0 = np.broadcast_to((20.0 - 0.2*(lat2[None,:]+60.0))[..., None], (nx, ny, nz)).copy()
S0 = np.full_like(T0, 35.0)
dt = 120.0
box = (-6.0, 42.0, 30.0, 46.5)
step, init_state, _, par, _ = make_solver_global(
    grid, physics, dt, forcing=(tau_x, tau_y, Q), eos_type='linear',
    T_init=T0, S_init=S0, sponge_days=3.0, sponge_cells=8,
    return_params=True, eta_relax_days=30.0, eta_relax_box=box, eta_relax_buffer=1.0)
st = init_state(T0, S0)
lon = np.asarray(grid.lon)
i0 = int(np.argmin(np.abs(lon-16.5))); j0 = int(np.argmin(np.abs(lat2-37.5)))
eta0 = np.asarray(st.eta).copy(); eta0[i0, j0] = 1.0
st = JaxStateG(st.u, st.v, st.T, st.S, jnp.array(eta0))
rate = 1.0/(30.0*86400.0)
area = np.asarray(par.dx_2d)*np.asarray(par.dy); wm = np.asarray(par.wet_mask)
A_ocean = float(np.sum(area*wm))
print(f"bump at ({lon[i0]},{lat2[j0]})  rate={rate:.6e}  A_ocean={A_ocean:.4e}")
sched = [(0,0),(100,100),(400,300),(800,400)]
for n, nrun in sched:
    if nrun:
        for _ in range(nrun):
            st = step(st)
    e = float(np.asarray(st.eta)[i0, j0])
    gm = float(np.sum(area*np.asarray(st.eta))/A_ocean)
    theory = 1.0*np.exp(-2*rate*n*dt)   # 2 half-steps per full step
    print(f"step {n:5d}: eta_bump={e:.6f}  theory={theory:.6f}  gm_eta={gm:+.3e}")
print("BUMP_TEST_DONE")
