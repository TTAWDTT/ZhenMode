"""Is the 3D divergence in the advecting velocity the free-surface signal,
or a mode-split projection inconsistency?

Fz[0] = column-integrated horizontal divergence of the 3D (u,v).
If the free surface absorbs it:  Fz[0] == -d(eta)/dt  == div of the barotropic
velocity times depth.  Compare Fz[0] against the barotropic divergence built
from the depth-averaged velocity with the SAME discrete divergence operator.
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
T0 = np.asarray(d["T"], np.float64); S0 = np.asarray(d["S"], np.float64)
u0 = np.asarray(d["u"], np.float64); v0 = np.asarray(d["v"], np.float64)
e0 = np.asarray(d["eta"], np.float64)
init = np.load("init_fields_g360x120.npz")
T_atm_np = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])

phys = replace(PhysicsConfig(), nu_h=5e6, nu_bi=2e14, kappa_bi=2e14,
               kappa_gm=1000.0, kappa_redi=1000.0, kappa_v=1e-5,
               kappa_conv=0.05, gm_slope_max=0.005)
step, _, _, p, terms_fn = JS.make_solver_global(
    g, phys, 3600.0, T_atm=jnp.asarray(T_atm_np), lambda_bulk=BULK_LAMBDA_DEFAULT,
    mode_split=True, dt_bt=300.0, return_params=True)

st = JS.JaxStateG(u=jnp.asarray(u0), v=jnp.asarray(v0), T=jnp.asarray(T0),
                  S=jnp.asarray(S0), eta=jnp.asarray(e0))
wet3 = np.asarray(p.wet_mask_z, float)
surf = wet3[:, :, 0] > 0.5
DZN = np.asarray(p.dz_node).ravel()

Fz = np.asarray(JS._vertical_transport_iface(st.u, st.v, p))
Fz0 = Fz[:, :, 0]

# barotropic velocity (depth average) and its 2D divergence
ubt, vbt = JS._barotropic_velocity(st.u, st.v, p)
divbt = np.asarray(JS._divergence_conservative(ubt, vbt, p))
H = np.asarray(p.H) if hasattr(p, "H") else None

print("Fz[0]            rms %.4e  min %+.4e  max %+.4e" % (np.sqrt((Fz0[surf]**2).mean()), Fz0[surf].min(), Fz0[surf].max()))
print("div(ubt,vbt)     rms %.4e  min %+.4e  max %+.4e" % (np.sqrt((divbt[surf]**2).mean()), divbt[surf].min(), divbt[surf].max()))
if H is not None:
    print("H (depth)        rms %.4e  min %+.4e  max %+.4e" % (np.sqrt((np.asarray(H)[surf]**2).mean()), np.asarray(H)[surf].min(), np.asarray(H)[surf].max()))
    prod = divbt * np.asarray(H)
    print("div(ubt)*H       rms %.4e" % np.sqrt((prod[surf]**2).mean()))
    r = np.corrcoef(prod[surf], Fz0[surf])[0, 1]
    print("corr(div(ubt)*H, Fz[0]) = %+.4f" % r)
    print("ratio mean(div_bt*H)/mean(Fz0) = %+.4f" % (prod[surf].mean() / Fz0[surf].mean()))

# direct: is the 3D divergence consistent with an independent operator?
divh = np.asarray(JS._divergence_h(st.u, st.v, p))
colsum = (divh * DZN[None, None, :]).sum(axis=2)
print("\nsum(div_h*dz) vs Fz[0]: max abs diff %.4e" % np.abs(colsum - Fz0)[surf].max())

# Now the trace: does Fz[0] equal the eta tendency of one step?
s1 = step(st); jax.block_until_ready(s1)
deta = np.asarray(s1.eta) - np.asarray(st.eta)
print("\nSIGN / MAGNITUDE TEST (per unit area, m/s):")
print("  deta/dt          rms %.4e  min %+.4e  max %+.4e" % (
    np.sqrt((deta[surf] / p.dt)**2).mean() * 0 + np.sqrt(((deta[surf]/float(p.dt))**2).mean()),
    (deta[surf]/float(p.dt)).min(), (deta[surf]/float(p.dt)).max()))
print("  -Fz[0]           rms %.4e" % np.sqrt(((-Fz0[surf])**2).mean()))
print("  corr(deta/dt, -Fz[0]) = %+.4f" % np.corrcoef(deta[surf]/float(p.dt), -Fz0[surf])[0, 1])

# the eta at ckpt is in metres; the free-surface eqn is deta/dt = -div(H ubt)
if H is not None:
    Hn = np.asarray(p.H)
    print("\n  H*div(ubt) rms %.4e" % np.sqrt(((Hn[surf] * divbt[surf]) ** 2).mean()))
    print("  corr(H*div(ubt), deta/dt) = %+.4f"
          % np.corrcoef(Hn[surf] * divbt[surf], deta[surf] / float(p.dt))[0, 1])
print("\nmax |eta| = %.4f m ; dz[0] = %.1f m ; ratio %.3f" %
      (np.abs(e0[surf]).max(), DZN[0], np.abs(e0[surf]).max() / DZN[0]))
