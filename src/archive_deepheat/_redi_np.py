"""Numpy re-implementation of the GM/Redi vertical skew-flux operator,
evaluated on the local 10-yr run snapshots, to attribute the deep warming.

Mirrors src/jax_solver_global.py:
  _isopycnal_slope (1160-1187), _redi_skew_flux_tendency (1251-1340),
  _d_dz (582), _gradient_conservative_3d (303), _divergence_conservative_3d (337),
  _fill_ghost_bottom (558).
"""
import glob
import numpy as np

RHO_0, ALPHA_T, BETA_S = 1025.0, 2.0e-4, 7.6e-4
C_P, T_REF, S_REF = 3992.0, 15.0, 35.0
GM_RHOZ_FLOOR = 1.0e-5
REDI_CFL_TARGET = 0.4
DT = 3600.0

Z = np.array([0, -5, -15, -30, -50, -75, -100, -150, -200, -300, -500, -1000, -2000, -4000], float)
DZ_NODE = np.concatenate([[abs(Z[1]-Z[0])],
                          0.5*(np.abs(np.diff(Z))[:-1] + np.abs(np.diff(Z))[1:]),
                          [abs(Z[-1]-Z[-2])]])
DZ_IFACE = np.abs(np.diff(Z))
DZ_DENOM = (np.abs(Z[2:]-Z[1:-1]) + np.abs(Z[1:-1]-Z[:-2]))   # interior, len nz-2
DZ_BND_TOP = float(abs(Z[1]-Z[0]))
DZ_BND_BOT = float(abs(Z[-1]-Z[-2]))


def fill_ghost_bottom(u, wet):
    idx = (wet > 0.5) * np.arange(u.shape[-1])
    kbot = idx.max(axis=-1)[..., None]
    kk = np.arange(u.shape[-1])[None, None, :]
    u_bot = np.take_along_axis(u, kbot, axis=-1)
    return np.where(kk >= kbot, u_bot, u)


def d_dz(u):
    du_top = (u[..., 1:2] - u[..., 0:1]) / DZ_BND_TOP
    du_int = (u[..., 2:] - u[..., :-2]) / DZ_DENOM
    du_bot = (u[..., -1:] - u[..., -2:-1]) / DZ_BND_BOT
    return np.concatenate([du_top, du_int, du_bot], axis=-1)


def gradient_conservative_3d(field, wet, cos_lat, inv_dx, inv_dy):
    open_xp = wet * np.roll(wet, -1, axis=0)
    open_xm = wet * np.roll(wet, 1, axis=0)
    gx = inv_dx * 0.5 * ((np.roll(field, -1, axis=0) - field) * open_xp
                         + (field - np.roll(field, 1, axis=0)) * open_xm)
    open_yp = wet * np.roll(wet, -1, axis=1)
    open_ym = wet * np.roll(wet, 1, axis=1)
    open_yp[:, -1, :] = 0.0
    open_ym[:, 0, :] = 0.0
    cfp = 0.5*(cos_lat + np.roll(cos_lat, -1))
    cfm = 0.5*(cos_lat + np.roll(cos_lat, 1))
    gy = (inv_dy / cos_lat[None, :, None]) * 0.5 * (
        (np.roll(field, -1, axis=1) - field) * open_yp * cfp[None, :, None]
        + (field - np.roll(field, 1, axis=1)) * open_ym * cfm[None, :, None])
    return gx, gy


def divergence_conservative_3d(Fx, Fy, wet, cos_lat, inv_dx, inv_dy):
    uo = wet * np.roll(wet, -1, axis=0)
    uf = 0.5*(Fx + np.roll(Fx, -1, axis=0)) * uo
    div_x = (uf - np.roll(uf, 1, axis=0)) * inv_dx
    vo = wet * np.roll(wet, -1, axis=1)
    vo[:, -1, :] = 0.0
    cfp = 0.5*(cos_lat + np.roll(cos_lat, -1))
    cfm = 0.5*(cos_lat + np.roll(cos_lat, 1))
    vf = 0.5*(Fy + np.roll(Fy, -1, axis=1)) * vo * cfp[None, :, None]
    div_y = (vf - np.roll(vf, 1, axis=1)) * (inv_dy / cos_lat[None, :, None])
    return div_x + div_y


def isopycnal_slope(T, S, wet, cos_lat, inv_dx, inv_dy, gm_slope_max):
    rho = RHO_0 * (-ALPHA_T*(T - T_REF) + BETA_S*(S - S_REF))
    rho = rho * wet
    rho_fill = fill_ghost_bottom(rho, wet)
    drho_dx, drho_dy = gradient_conservative_3d(rho, wet, cos_lat, inv_dx, inv_dy)
    drho_dz = d_dz(rho_fill)
    denom = np.where(np.abs(drho_dz) < GM_RHOZ_FLOOR,
                     np.sign(drho_dz)*GM_RHOZ_FLOOR + 1e-30, drho_dz)
    Sx = -drho_dx/denom
    Sy = -drho_dy/denom
    S2 = Sx*Sx + Sy*Sy
    s4 = (gm_slope_max**2)**2
    sigma = 1.0/(1.0 + (S2*S2)/s4)
    Sx_t = sigma*Sx
    Sy_t = sigma*Sy
    S2_t = Sx_t*Sx_t + Sy_t*Sy_t
    return Sx_t*wet, Sy_t*wet, S2_t, Sx, Sy, S2


def redi_vertical_only(tracer, Sx, Sy, wet, kappa, cos_lat, inv_dx, inv_dy):
    """Just the vertical part of Fz_i: the S2 * dC_dz_iface term."""
    Cm = tracer[..., :-1]; Cp = tracer[..., 1:]
    dCdz_i = (Cp - Cm)/DZ_IFACE
    Sx_i = 0.5*(Sx[..., :-1] + Sx[..., 1:])
    Sy_i = 0.5*(Sy[..., :-1] + Sy[..., 1:])
    S2_i = Sx_i*Sx_i + Sy_i*Sy_i
    dzu = DZ_NODE[:-1]; dzl = DZ_NODE[1:]
    Dv_max = REDI_CFL_TARGET*DZ_IFACE/(DT*(1.0/dzu + 1.0/dzl))
    S2_eff = np.minimum(S2_i, Dv_max/kappa)
    Fz = -kappa*(S2_eff*dCdz_i)
    # mask: zero flux where either side dry
    live = (wet[..., :-1] > 0.5) & (wet[..., 1:] > 0.5)
    Fz = np.where(live, Fz, 0.0)
    up = np.concatenate([np.zeros_like(Fz[..., :1]), Fz], axis=-1)
    dn = np.concatenate([Fz, np.zeros_like(Fz[..., :1])], axis=-1)
    return (up - dn)/DZ_NODE[None, None, :]*wet, S2_eff*kappa


def main():
    snaps = sorted(glob.glob("results/global_tenyr_ms_gm_3d/*.npy"))
    print("snaps", len(snaps))
    # grid: rebuild with the project's own grid module (numpy parts only)
    import sys; sys.path.insert(0, "src")
    from grid import make_global_grid, GlobalGridConfig
    from config import DEFAULT_CONFIG
    g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                         DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
    AREA = np.asarray(g.dx_2d, float)*float(g.dy)
    cos_lat = np.asarray(g.cos_lat, float)
    inv_dx = 1.0/np.asarray(g.dx_2d, float)[..., None]
    inv_dy = 1.0/float(g.dy)
    print("grid z:", np.asarray(g.z))
    print("wet_mask_3d sum:", int(np.asarray(g.wet_mask_3d).sum()))
    print("AREA", AREA.shape, "cos_lat", cos_lat.shape, "inv_dx", inv_dx.shape)

    snap = np.load(snaps[-1])
    T = snap[0]; S = snap[3]
    ghost = np.abs(T - 15.0) < 1e-9
    wet = (~ghost).astype(float)
    print("wet3", int(wet.sum()))

    for gm_slope_max, kappa in ((0.005, 1000.0),):
        Sx, Sy, S2_t, Sx_r, Sy_r, S2_r = isopycnal_slope(
            T, S, wet, cos_lat, inv_dx, inv_dy, gm_slope_max)
        # raw slope magnitudes at depth
        print()
        print("=== raw |S| vs tapered |S|, kappa=%.0f slope_max=%.4f ===" % (kappa, gm_slope_max))
        for k in (0, 3, 6, 8, 10, 11, 12, 13):
            m = wet[:, :, k] > 0.5
            print("  k=%2d z=%6.0f  |S_raw| mean %.4f max %.4f | |S_tap| mean %.4f max %.4f"
                  % (k, Z[k], np.sqrt(S2_r[:, :, k][m]).mean(), np.sqrt(S2_r[:, :, k][m]).max(),
                     np.sqrt(S2_t[:, :, k][m]).mean(), np.sqrt(S2_t[:, :, k][m]).max()))
        tend_v, Dv = redi_vertical_only(T, Sx, Sy, wet, kappa, cos_lat, inv_dx, inv_dy)
        print()
        print("=== implied vertical diffusivity D_v = kappa*|S_tap|^2 [m2/s] (on interfaces) ===")
        for k in range(13):
            m = (wet[:, :, k] > 0.5) & (wet[:, :, k+1] > 0.5)
            zc = 0.5*(Z[k]+Z[k+1])
            print("  iface %2d z=%6.0f  D_v mean %.5f  p99 %.5f  max %.5f   (ratio to kappa_v=1e-5: %4.0fx / %4.0fx)"
                  % (k, zc, Dv[:, :, k][m].mean(), np.percentile(Dv[:, :, k][m], 99),
                     Dv[:, :, k][m].max(), Dv[:, :, k][m].mean()/1e-5, np.percentile(Dv[:, :, k][m],99)/1e-5))
        vol = wet*AREA[:, :, None]*DZ_NODE[None, None, :]
        RC = RHO_0*C_P
        r = (tend_v*vol).sum(axis=(0, 1))*RC*3.15576e7/1e21
        print()
        print("=== OHC tendency from Redi VERTICAL term only [ZJ/yr per level] ===")
        for k in range(14):
            print("  k=%2d z=%6.0f  %+8.3f ZJ/yr" % (k, Z[k], r[k]))
        print("  TOTAL %+8.3f ZJ/yr" % r.sum())


if __name__ == "__main__":
    main()
