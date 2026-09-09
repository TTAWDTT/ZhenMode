# -*- coding: utf-8 -*-
"""Spinup dashboard analysis (cluster-side, self-contained) — v3.

Reads results/global_<TAG>.npz (2D tables, written at completion) and the
yearly 3D snaps in results/global_<TAG>_3d/, and computes ~18 per-snapshot
series for the dashboard plus the three spinup_plan_zh.md probe criteria:

circulation : AMOC(30-60N), AMOC@26N, AABW cell, Pacific overturning,
              Drake-passing transport (partial section), total KE
tracers     : SST global / NAtlantic, SSS global, deep S (z<=1000),
              T mean 0-2000 m, deepT global / Atlantic, OHC (ZJ),
              upper-ocean stratification (dT/dz, 0-100 m)
stability   : max|u|, max|eta| (npz table, completion only), SSH std
profiles    : Atlantic T(z)/S(z) + global T(z)/S(z) (last snap, 1-D), and
              AMOC ψ(y,z) section (last snap, flattened 2-D, row-major
              lat-major for the frontend)

Criteria (v3 adds d/e/f and sanity warnings):
  a no-blowup / b deepT trend / c Med eta     (unchanged)
  d OHC trend        |ZJ/yr| over yr2..end, |slope| < 5 PASS (target <2)
  e SSS drift        psu/yr over yr2..end, |slope| < 0.003 PASS
  f AMOC sanity      flags absolute-value problems:
                     amoc_sane (>25 Sv warning) + amoc26_ratio (26N/30-60N
                     collapsed <0.4 warning) — information badges, not pass/fail

Deep-T uses the 3-D layer-resolved wet mask (wet3_push.npz, rebuilt from
ETOPO via the grid.py pipeline) so below-floor ghost fill is excluded.
Writes results/<TAG>_analysis.json (+ .npz) and prints a verdict.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = "/data/tmp/ocean"
TAG = sys.argv[1] if len(sys.argv) > 1 else "spinA30probe"
NPZ = os.path.join(ROOT, "results", f"global_{TAG}.npz")
D3 = os.path.join(ROOT, "results", f"global_{TAG}_3d")
W3 = os.path.join(ROOT, "results", "wet3_push.npz")
OUT_JSON = os.path.join(ROOT, "results", f"{TAG}_analysis.json")
OUT_NPZ = os.path.join(ROOT, "results", f"{TAG}_analysis.npz")

R_EARTH = 6.371e6
RHO_CP = 1025.0 * 3990.0
BASELINE_DEEPT_C_PER_YR = 0.045   # kappa_v=1e-5 10-yr run (moc_tenyr_ms_gm)
OHC_TREND_WARN_ZJ_YR = 5.0        # criteria d: |ZJ/yr| budget closure
OHC_TREND_PASS_ZJ_YR = 2.0
SSS_TREND_PASS_PSU_YR = 0.003     # criteria e: drift target ~3x SSS flux
AMOC_WARN_SV = 25.0               # criteria f: absolute-value sanity
AMOC26_RATIO_WARN = 0.4

snap_files = sorted(glob.glob(os.path.join(D3, "snap_*.npy")))
n = len(snap_files)
days = np.array([int(os.path.basename(f)[5:10]) * 365.0
                 for f in snap_files])
if n == 0:
    print("no 3D snaps yet; nothing to do")
    sys.exit(0)

d = None
if os.path.exists(NPZ):
    d = np.load(NPZ, allow_pickle=True)
else:
    # mid-run: final npz not yet written — proceed with 3D snaps only
    print(f"NOTE: {NPZ} not found (run still in progress); "
          "verdict/max-eta from tables unavailable")

# Day axis from the snap FILENAME index (snap_00029 -> day 29*365), which is
# correct for both fresh runs (index 0..N) and --restart-from resumes (index
# starts at n_prev_snaps). The runner npz `days` table is NOT usable mid-run:
# it only exists at completion, and a stale same-tag npz from an earlier era
# would silently mismatch the snap count.
# npz verdict/max-eta are only valid when the table was written by THIS run
# (final npz) — gate on last-day agreement to reject stale same-tag npz from
# an earlier era (table row count can legitimately differ from snap-file count
# on --restart-from resumes: the parent era's boundary snaps stay on disk).
npz_ok = (d is not None and len(d["days"]) > 0
          and abs(float(d["days"][-1]) - days[-1]) < 1.0)
verdict = str(d["verdict"]) if npz_ok else "RUNNING"
max_eta_all = float(np.max(np.abs(d["max_eta"]))) if npz_ok else float("nan")
sshstd_last = (float(d["ssh_std"][-1]) if npz_ok and "ssh_std" in d.files
               else float("nan"))
# Parent-era max|eta| (same integration, pre-resume segment): the era-2 table
# starts at the resume day, so set PARENT_NPZ to merge it in.
PARENT_NPZ = os.environ.get("PARENT_NPZ", "")
if npz_ok and PARENT_NPZ and os.path.exists(PARENT_NPZ):
    dp = np.load(PARENT_NPZ, allow_pickle=True)
    if len(dp["max_eta"]):
        max_eta_all = max(max_eta_all,
                          float(np.max(np.abs(dp["max_eta"]))))
max_u_peak = float(d["max_u_peak"]) if d else float("nan")

lat = d["lat"] if (d and npz_ok) else np.linspace(-59.5, 59.5, 120)
lon = d["lon"] if (d and npz_ok) else (np.arange(360) + 0.5)
z = d["z"] if (d and npz_ok) else np.array(
    [0, -5, -15, -30, -50, -75, -100, -150, -200, -300, -500, -1000, -2000,
     -4000], dtype=float)

# ── masks & geometry ──
nlon, nlat = lon.size, lat.size
atl_i = np.zeros(nlon, dtype=bool)
atl_i[(lon >= 280.0)] = True
atl_i[(lon < 25.0)] = True
atl2 = np.broadcast_to(atl_i[:, None], (nlon, nlat)).copy()
pac_i = (lon >= 130.0) & (lon < 280.0)
pac2 = np.broadcast_to(pac_i[:, None], (nlon, nlat)).copy()
i_drake = int(np.argmin(np.abs(lon - 291.5)))   # ~68.5 W, Drake Passage

w = np.load(W3)
wet3 = (w["wet3"] if "wet3" in w.files else w[w.files[0]]) > 0.5
wet3_deep = wet3 & (z <= -1000.0)[None, None, :]
wet3_deep_atl = wet3_deep & atl_i[:, None, None]
wet3_2k = wet3 & (z >= -2000.0)[None, None, :]
wet3_deep_pac = wet3_deep & pac_i[:, None, None]
wet3_col = wet3[:, :, 0:1]                      # column mask for MOC

dlat = float(np.median(np.abs(np.diff(lat))))    # actual grid spacing (deg)
dlon = float(np.median(np.abs(np.diff(lon))))
dy = R_EARTH * np.radians(dlat)                  # 1 deg → 111.2 km
dx = R_EARTH * np.radians(dlon) * np.cos(np.deg2rad(lat))
dz = np.empty(z.size)
dz[0] = z[0] - z[1]
dz[-1] = z[-2] - z[-1]
for k in range(1, z.size - 1):
    dz[k] = 0.5 * (z[k - 1] - z[k + 1])
V3 = dx[None, :, None] * dy * dz[None, None, :]   # cell volumes (1,ny,nz)


def moc_psi(v, lons=None):
    if lons is None:
        vI = np.where(wet3_col, v, 0.0)
    else:
        vI = np.where(lons[:, :, None] & wet3_col, v, 0.0)
    V = vI.sum(axis=0) * dx[:, None] * dz[None, :]
    return np.cumsum(V, axis=1) / 1.0e6          # Sv


jband = (lat >= 30.0) & (lat <= 60.0)
j26 = (lat >= 25.0) & (lat <= 28.0)
jabw = (lat >= -60.0) & (lat <= -30.0)
kbnd = (z <= -300.0) & (z >= -3000.0)
kabw = z <= -2000.0
kpac = (z <= -300.0) & (z >= -3000.0)
jpac = (lat >= 0.0) & (lat <= 60.0)
surf2d = wet3[:, :, 0]
natl2d = surf2d & atl2 & ((lat >= 10.0) & (lat <= 60.0))[None, :]

print(f"{n} 3D snaps, days {days[0]:.0f}..{days[-1]:.0f}, verdict={verdict}")

S = dict((k, np.zeros(n)) for k in
         ["amoc", "amoc_26n", "aabw", "pmoc", "drake", "ke",
          "sst", "sst_natl", "sss", "deepS", "t2k", "deepT", "deepT_atl",
          "ohc", "strat", "maxu"])
for i, f in enumerate(snap_files):
    s = np.load(f)                    # (4, nx, ny, nz) = T,u,v,S
    T, u, v, Sfld = s[0], s[1], s[2], s[3]

    if i == n - 1:                    # keep last snap for profiles/section
        T_last, S_last, v_last = T, Sfld, v

    psi_a = moc_psi(v, lons=atl2)
    S["amoc"][i] = psi_a[np.ix_(jband, kbnd)].max()
    S["amoc_26n"][i] = psi_a[np.ix_(j26, kbnd)].max()
    S["aabw"][i] = psi_a[np.ix_(jabw, kabw)].min()

    psi_p = moc_psi(v, lons=pac2)
    S["pmoc"][i] = psi_p[np.ix_(jpac, kpac)].max()

    # Drake Passage section: fixed lon (axis 0), integrate meridionally
    uI = np.where(wet3[i_drake], u[i_drake], 0.0)
    S["drake"][i] = (uI.sum(axis=0) * dy * dz).sum() / 1.0e6

    ke_cell = np.broadcast_to(V3, wet3.shape)      # (nx,ny,nz) cell volumes
    S["ke"][i] = 0.5 * 1025.0 * float(
        np.sum((u * u + v * v)[wet3] * ke_cell[wet3])) / 1.0e18  # EJ
    S["ohc"][i] = RHO_CP * float(
        np.sum(T[wet3] * ke_cell[wet3])) / 1.0e21  # ZJ

    S["maxu"][i] = float(np.max(np.abs(u)))
    S["sst"][i] = float(np.mean(T[:, :, 0][surf2d]))
    S["sst_natl"][i] = float(np.mean(T[:, :, 0][natl2d]))
    S["sss"][i] = float(np.mean(Sfld[:, :, 0][surf2d]))
    S["deepS"][i] = float(np.mean(Sfld[wet3_deep]))
    S["t2k"][i] = float(np.mean(T[wet3_2k]))
    S["deepT"][i] = float(np.mean(T[wet3_deep]))
    S["deepT_atl"][i] = float(np.mean(T[wet3_deep_atl]))
    dTdz = (T[:, :, 0] - T[:, :, 6]) / 100.0        # 0 -> -100 m
    S["strat"][i] = float(np.mean(dTdz[surf2d]))
    del s, T, u, v, Sfld

yrs = days / 365.0
# linear trend over yr2..end (skip yr0-1 spinup transient)
m = yrs >= 2.0
if m.sum() >= 3:
    slope_g = float(np.polyfit(yrs[m], S["deepT"][m], 1)[0])
    slope_a = float(np.polyfit(yrs[m], S["deepT_atl"][m], 1)[0])
    # v3: budget-closure criteria — OHC trend (ZJ/yr) and SSS drift (psu/yr)
    slope_ohc = float(np.polyfit(yrs[m], S["ohc"][m], 1)[0])
    slope_sss = float(np.polyfit(yrs[m], S["sss"][m], 1)[0])
else:
    slope_g = slope_a = slope_ohc = slope_sss = float("nan")

# ── profiles (last snap): Atlantic & global T(z)/S(z), wet-points only ──
prof_t_atl = [float(np.mean(T_last[:, :, k][wet3[:, :, k] & atl2]))
              for k in range(z.size)]
prof_s_atl = [float(np.mean(S_last[:, :, k][wet3[:, :, k] & atl2]))
              for k in range(z.size)]
prof_t_glb = [float(np.mean(T_last[:, :, k][wet3[:, :, k]]))
              for k in range(z.size)]
prof_s_glb = [float(np.mean(S_last[:, :, k][wet3[:, :, k]]))
              for k in range(z.size)]

# ── AMOC ψ(y,z) section (last snap, Atlantic basin), row-major lat-major:
#    psi[i*nz + k] = psi(lat[i], z[k]) — frontend reshapes to (nlat, nz)
psi_last = moc_psi(v_last, lons=atl2)          # (nlat, nz) Sv
psi_flat = psi_last.astype(float).ravel().tolist()
psi_shape = [int(psi_last.shape[0]), int(psi_last.shape[1])]

m2 = yrs >= 2.0
amoc_mean = float(S["amoc"][m2].mean()) if m2.sum() else float("nan")
amoc_last5 = (float(S["amoc"][yrs >= yrs[-1] - 5.0].mean())
              if n >= 2 else float("nan"))

# v3 sanity: real-ocean AMOC ~14-20 Sv; ψ>25 Sv suggests the basin/units
# diagnostic (not the physics) is off; AMOC@26N collapsing to <40% of the
# 30-60N max is the shutdown signature seen in kv1e5.
amoc26_ratio = (float(S["amoc_26n"][-1] / S["amoc"][-1])
                if S["amoc"][-1] != 0 else float("nan"))
warnings = []
if np.isfinite(S["amoc"][-1]) and abs(S["amoc"][-1]) > AMOC_WARN_SV:
    warnings.append(f"AMOC 30-60N {S['amoc'][-1]:.1f} Sv > {AMOC_WARN_SV:.0f} "
                    "(absolute value suspect)")
if np.isfinite(amoc26_ratio) and amoc26_ratio < AMOC26_RATIO_WARN:
    warnings.append(f"AMOC@26N / 30-60N = {amoc26_ratio:.2f} < "
                    f"{AMOC26_RATIO_WARN:.1f} (collapse signature)")

crit = {
    "a_no_blowup": verdict == "PASS" if npz_ok else None,
    "b_deept_trend": {
        "global_C_per_yr": slope_g,
        "atlantic_C_per_yr": slope_a,
        "baseline_C_per_yr": BASELINE_DEEPT_C_PER_YR,
        "reduction_factor": (BASELINE_DEEPT_C_PER_YR / slope_g
                             if slope_g and slope_g > 0 else None),
        "pass": bool(slope_g < 0.5 * BASELINE_DEEPT_C_PER_YR),
    },
    "c_med_eta": {
        "max_eta_m": max_eta_all,
        "watchdog_m": 15.0,
        "pass": bool(max_eta_all < 15.0) if npz_ok else None,
    },
    "d_ohc_trend": {
        "ZJ_per_yr": slope_ohc,
        "warn_abs": OHC_TREND_WARN_ZJ_YR,
        "pass": bool(abs(slope_ohc) < OHC_TREND_PASS_ZJ_YR)
        if np.isfinite(slope_ohc) else None,
    },
    "e_sss_drift": {
        "psu_per_yr": slope_sss,
        "pass_abs": SSS_TREND_PASS_PSU_YR,
        "pass": bool(abs(slope_sss) < SSS_TREND_PASS_PSU_YR)
        if np.isfinite(slope_sss) else None,
    },
    "f_amoc_sanity": {
        "amoc_final_sv": float(S["amoc"][-1]),
        "amoc26_ratio": amoc26_ratio,
        "warn": warnings,
    },
}

out = {
    "tag": TAG,
    "verdict": verdict,
    "max_u_peak": max_u_peak,
    "n_snaps": n,
    "criteria": crit,
    "amoc": {
        "day0": float(S["amoc"][0]), "final": float(S["amoc"][-1]),
        "mean_yr2_end": amoc_mean, "last5yr_mean": amoc_last5,
    },
    "deepT": {
        "day0": float(S["deepT"][0]), "final": float(S["deepT"][-1]),
        "atl_final": float(S["deepT_atl"][-1]),
        "T2k_final": float(S["t2k"][-1]),
    },
    "warnings": warnings,
    "profiles": {
        "z": [float(v) for v in z],
        "T_atl": prof_t_atl, "S_atl": prof_s_atl,
        "T_glb": prof_t_glb, "S_glb": prof_s_glb,
    },
    "amoc_psi": {"shape": psi_shape, "psi": psi_flat,
                 "lat": [float(v) for v in lat],
                 "z": [float(v) for v in z]},
}
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)

# npz payload: all 1-D series (dashboard decoder keeps ndim==1 keys), plus the
# criteria flattened as 1-element arrays so the UI can badge them live.
npz_out = {k: v for k, v in S.items()}
npz_out.update(days=days, yrs=yrs, sshstd=np.full(n, sshstd_last)
               if np.isfinite(sshstd_last) else np.zeros(n),
               maxeta=np.full(n, max_eta_all) if npz_ok else np.zeros(n),
               crit_slope_g=np.array([slope_g]),
               crit_slope_a=np.array([slope_a]),
               crit_slope_ohc=np.array([slope_ohc]),
               crit_slope_sss=np.array([slope_sss]),
               crit_max_eta=np.array([max_eta_all]),
               crit_b_pass=np.array([1.0 if crit["b_deept_trend"]["pass"]
                                     else 0.0]),
               crit_c_pass=np.array([1.0 if crit["c_med_eta"]["pass"]
                                     else 0.0]),
               crit_d_pass=np.array([1.0 if crit["d_ohc_trend"]["pass"]
                                     else 0.0]),
               crit_e_pass=np.array([1.0 if crit["e_sss_drift"]["pass"]
                                     else 0.0]),
               amoc26_ratio=np.array([amoc26_ratio]),
               prof_z=np.array([float(v) for v in z]),
               prof_lat=np.array([float(v) for v in lat]),
               prof_T_atl=np.array(prof_t_atl),
               prof_S_atl=np.array(prof_s_atl),
               prof_T_glb=np.array(prof_t_glb),
               prof_S_glb=np.array(prof_s_glb))
# ψ section is 2-D — flatten lat-major; frontend reshapes by shape=[nlat,nz]
np.savez(OUT_NPZ, **npz_out,
         psi_flat=np.array(psi_flat), psi_shape=np.array(psi_shape))

print(json.dumps(out, indent=2))
print(f"saved {OUT_JSON}")
print(f"saved {OUT_NPZ}")
