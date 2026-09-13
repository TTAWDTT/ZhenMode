"""Mass-budget closure of the advecting velocity, interface by interface.

For a fixed-layer z-coordinate model the mass budget of every control volume
must close.  Check sum(area * Fz[k]) at each interface and the heat carried,
area * Fz[k] * T_donor.  Anything nonzero at k=0 is heat exchanged with the
free surface; anything nonzero at k>0 means the discrete velocity is not
divergence-free.
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
d = np.load("results/ckpt_tenyr_ms_gm.npz")
init = np.load("init_fields_g360x120.npz")
T_atm_np = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])

phys = replace(PhysicsConfig(), nu_h=5e6, nu_bi=2e14, kappa_bi=2e14,
               kappa_gm=1000.0, kappa_redi=1000.0, kappa_v=1e-5,
               kappa_conv=0.05, gm_slope_max=0.005)
_, _, _, p, _ = JS.make_solver_global(
    g, phys, 3600.0, T_atm=jnp.asarray(T_atm_np), lambda_bulk=BULK_LAMBDA_DEFAULT,
    mode_split=True, dt_bt=300.0, return_params=True)

st = JS.JaxStateG(u=jnp.asarray(d["u"], np.float64), v=jnp.asarray(d["v"], np.float64),
                  T=jnp.asarray(d["T"], np.float64), S=jnp.asarray(d["S"], np.float64),
                  eta=jnp.asarray(d["eta"], np.float64))
wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
surf = wet3[:, :, 0] > 0.5
H = np.asarray(g.depth, float)

Fz = np.asarray(JS._vertical_transport_iface(st.u, st.v, p))
T = np.asarray(st.T)
Tdeep = np.asarray(JS._fill_ghost_bottom(st.T, p))[..., 1:]

CPR = 1025.0 * 3992.0 / 1e21 * 3.1536e7   # -> ZJ/yr per (m^3/s * K)

print("interface mass + heat budget   (Fz area integral m^3/s ; heat ZJ/yr)")
print("  k    z_top    sum(A*Fz)      sum|A*Fz|     sum(A*Fz*Tdon)   n_active")
Ztop = np.array([0, 5, 15, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, 4000], float)
tot_m = 0.0
for k in range(14):
    f = Fz[:, :, k]
    a = AREA * (wet3[:, :, k] > 0.5)
    m = float((a * f).sum())
    ma = float((a * np.abs(f)).sum())
    if k < 13:
        tdon = np.where(Fz[:, :, k + 1] > 0.0, T[:, :, k], Tdeep[:, :, k])
        th = float((a * f * tdon).sum() * CPR)
    else:
        th = 0.0
    na = int((np.abs(f) > 0).sum())
    print("  %2d  %-8.0f %+12.4e  %12.4e  %+12.3f      %6d" % (k, Ztop[k], m, ma, th, na))

print("\nsum(A*Fz) over all interfaces = %+.4e m^3/s (each should be ~0)"
      % sum(float((AREA * (wet3[:, :, k] > 0.5) * Fz[:, :, k]).sum()) for k in range(14)))

# horizontal divergence closure per level
divh = np.asarray(JS._divergence_h(st.u, st.v, p))
print("\nper-level sum(A*div_h) (m^3/s, should be 0 with closed walls):")
for k in range(14):
    a = AREA * (wet3[:, :, k] > 0.5)
    print("  k=%2d  %+12.4e" % (k, float((a * divh[:, :, k]).sum())))

# eta consistency
step, _, _, _, _ = JS.make_solver_global(
    g, phys, 3600.0, T_atm=jnp.asarray(T_atm_np), lambda_bulk=BULK_LAMBDA_DEFAULT,
    mode_split=True, dt_bt=300.0, return_params=True)
s1 = step(st)
deta = (np.asarray(s1.eta) - np.asarray(st.eta)) / float(p.dt)
print("\nfree-surface consistency:")
print("  rms Fz[0]/H          %.4e  (expected -deta/dt)" % np.sqrt(((Fz[:, :, 0] / H)[surf] ** 2).mean()))
print("  rms deta/dt          %.4e" % np.sqrt((deta[surf] ** 2).mean()))
print("  corr                 %+.4f" % np.corrcoef((Fz[:, :, 0] / H)[surf], deta[surf])[0, 1])
